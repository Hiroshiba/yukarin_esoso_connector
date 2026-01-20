import json
from pathlib import Path

import numpy
import torch
import yaml
from g2p_en import G2p
from hifi_gan.env import AttrDict
from hifi_gan.models import Generator as HifiGanGenerator
from upath import UPath
from yukarin_es.config import Config as ConfigEs
from yukarin_es.generator import Generator as GeneratorEs
from yukarin_es.utility.upath_utility import to_local_path
from yukarin_esad.config import Config as ConfigEsad
from yukarin_esad.generator import Generator as GeneratorEsad
from yukarin_esosoad.config import Config as ConfigEsosoad
from yukarin_esosoad.generator import Generator as GeneratorEsosoad

from .utility import get_predictor_model_path, remove_weight_norm

ARPA_PHONEMES = [
    "pau",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
    "B",
    "CH",
    "D",
    "DH",
    "F",
    "G",
    "HH",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "spn",
]

VOWEL_PHONEMES = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}


def _ensure_nltk_data():
    import nltk

    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    try:
        nltk.data.find("corpora/cmudict")
    except LookupError:
        nltk.download("cmudict", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def text_to_phoneme_data(text: str) -> tuple[list[str], list[int], list[int]]:
    _ensure_nltk_data()
    g2p = G2p()
    phonemes_raw = g2p(text)

    phonemes = []
    stress_list = []

    for ph in phonemes_raw:
        if ph == " ":
            continue
        elif ph[-1] in ("0", "1", "2"):
            phonemes.append(ph[:-1])
            stress_list.append(int(ph[-1]))
        elif ph.upper() in ARPA_PHONEMES:
            phonemes.append(ph.upper())
            stress_list.append(0)
        elif ph in (",", ".", "!", "?", ";", ":", "-"):
            phonemes.append("pau")
            stress_list.append(0)
        else:
            raise ValueError(f"Unknown phoneme: {ph}")

    if not phonemes:
        phonemes = ["pau"]
        stress_list = [0]

    if phonemes[0] != "pau":
        phonemes.insert(0, "pau")
        stress_list.insert(0, 0)
    if phonemes[-1] != "pau":
        phonemes.append("pau")
        stress_list.append(0)

    vowel_indices = [i for i, ph in enumerate(phonemes) if ph in VOWEL_PHONEMES]

    return phonemes, stress_list, vowel_indices


def text_to_word_phoneme_data(
    text: str,
) -> tuple[list[str], list[str], list[int], list[int], list[int]]:
    _ensure_nltk_data()

    import re

    text_words = re.findall(r"\w+", text)

    g2p = G2p()
    phonemes_raw = g2p(text)

    words = []
    all_phonemes = []
    all_stress_list = []
    word_boundaries = []
    in_word = False
    word_idx = 0

    for ph in phonemes_raw:
        if ph == " ":
            if in_word:
                in_word = False
            continue
        elif ph[-1] in ("0", "1", "2"):
            if not in_word:
                word_boundaries.append(len(all_phonemes))
                words.append(text_words[word_idx])
                word_idx += 1
                in_word = True
            all_phonemes.append(ph[:-1])
            all_stress_list.append(int(ph[-1]))
        elif ph.upper() in ARPA_PHONEMES:
            if not in_word:
                word_boundaries.append(len(all_phonemes))
                words.append(text_words[word_idx])
                word_idx += 1
                in_word = True
            all_phonemes.append(ph.upper())
            all_stress_list.append(0)
        elif ph in (",", ".", "!", "?", ";", ":", "-"):
            if in_word:
                in_word = False
            word_boundaries.append(len(all_phonemes))
            words.append("pau")
            all_phonemes.append("pau")
            all_stress_list.append(0)
        else:
            raise ValueError(f"Unknown phoneme: {ph}")

    if not all_phonemes:
        all_phonemes = ["pau"]
        all_stress_list = [0]
        words = ["pau"]
        word_boundaries = [0, 1]
    else:
        if all_phonemes[0] != "pau":
            all_phonemes.insert(0, "pau")
            all_stress_list.insert(0, 0)
            word_boundaries = [b + 1 for b in word_boundaries]
            words.insert(0, "pau")
            word_boundaries.insert(0, 0)

        if all_phonemes[-1] != "pau":
            all_phonemes.append("pau")
            all_stress_list.append(0)
            words.append("pau")
            word_boundaries.append(len(all_phonemes) - 1)

        word_boundaries.append(len(all_phonemes))

    vowel_indices = [i for i, ph in enumerate(all_phonemes) if ph in VOWEL_PHONEMES]

    return words, all_phonemes, all_stress_list, vowel_indices, word_boundaries


class Forwarder:
    def __init__(
        self,
        yukarin_es_model_dir: UPath | Path,
        yukarin_esad_model_dir: UPath | Path,
        yukarin_esosoad_model_dir: UPath | Path,
        hifigan_model_path: UPath | Path,
        use_gpu: bool,
        yukarin_es_iteration: int | None = None,
        yukarin_esad_iteration: int | None = None,
        yukarin_esosoad_iteration: int | None = None,
    ):
        yukarin_es_model_dir = UPath(yukarin_es_model_dir)
        yukarin_esad_model_dir = UPath(yukarin_esad_model_dir)
        yukarin_esosoad_model_dir = UPath(yukarin_esosoad_model_dir)
        hifigan_model_path = UPath(hifigan_model_path)
        hifigan_config_path = hifigan_model_path.parent / "config.json"

        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        config_es = ConfigEs.from_dict(
            yaml.safe_load((yukarin_es_model_dir / "config.yaml").read_text())
        )

        predictor_es_path = get_predictor_model_path(
            yukarin_es_model_dir, iteration=yukarin_es_iteration
        )
        print("yukarin_es predictor:", predictor_es_path)

        self.yukarin_es_generator = GeneratorEs(
            config=config_es,
            predictor=to_local_path(predictor_es_path),
            use_gpu=use_gpu,
        )
        self.yukarin_es_generator.predictor.apply(remove_weight_norm)
        print("yukarin_es loaded!")

        config_esad = ConfigEsad.from_dict(
            yaml.safe_load((yukarin_esad_model_dir / "config.yaml").read_text())
        )

        predictor_esad_path = get_predictor_model_path(
            yukarin_esad_model_dir, iteration=yukarin_esad_iteration
        )
        print("yukarin_esad predictor:", predictor_esad_path)
        self.yukarin_esad_generator = GeneratorEsad(
            config=config_esad,
            predictor=to_local_path(predictor_esad_path),
            use_gpu=use_gpu,
        )
        self.yukarin_esad_generator.predictor.apply(remove_weight_norm)
        print("yukarin_esad loaded!")

        config_esosoad = ConfigEsosoad.from_dict(
            yaml.safe_load((yukarin_esosoad_model_dir / "config.yaml").read_text())
        )

        predictor_esosoad_path = get_predictor_model_path(
            yukarin_esosoad_model_dir, iteration=yukarin_esosoad_iteration
        )
        print("yukarin_esosoad predictor:", predictor_esosoad_path)
        self.yukarin_esosoad_generator = GeneratorEsosoad(
            config=config_esosoad,
            predictor=to_local_path(predictor_esosoad_path),
            use_gpu=use_gpu,
        )
        self.yukarin_esosoad_generator.predictor.apply(remove_weight_norm)
        print("yukarin_esosoad loaded!")

        hifigan_config = AttrDict(
            json.loads(to_local_path(hifigan_config_path).read_text())
        )
        self.hifigan_generator = HifiGanGenerator(hifigan_config).to(self.device)
        state_dict = torch.load(
            to_local_path(hifigan_model_path), map_location=self.device
        )
        self.hifigan_generator.load_state_dict(state_dict["generator"])
        self.hifigan_generator.eval()
        self.hifigan_generator.remove_weight_norm()
        self.sampling_rate = hifigan_config.sampling_rate
        self.hop_size = hifigan_config.hop_size
        print("hifi-gan loaded!")

        self.phoneme_to_id = {ph: i for i, ph in enumerate(ARPA_PHONEMES)}

    @torch.no_grad()
    def forward(
        self,
        text: str,
        speaker_id: int,
        f0_speaker_id: int | None = None,
        f0_correct: float = 0,
        diffusion_step_num: int = 10,
    ) -> tuple[numpy.ndarray, tuple]:
        if f0_speaker_id is None:
            f0_speaker_id = speaker_id

        phonemes, stress_list, vowel_indices = text_to_phoneme_data(text)
        print(f"Phonemes: {phonemes}")
        print(f"Stress: {stress_list}")
        print(f"Vowel indices: {vowel_indices}")

        phoneme_ids = numpy.array(
            [self.phoneme_to_id[ph] for ph in phonemes], dtype=numpy.int64
        )
        phoneme_ids_tensor = torch.from_numpy(phoneme_ids).to(self.device)

        duration_output = self.yukarin_es_generator(
            phoneme_id_list=[phoneme_ids_tensor],
            speaker_id=numpy.array([speaker_id]),
        )
        durations = duration_output.duration[0].cpu().numpy()
        durations[0] = durations[-1] = 0.08
        durations[durations < 0.01] = 0.01

        phoneme_ids_tensor_list = [phoneme_ids_tensor]
        durations_tensor_list = [torch.from_numpy(durations).float().to(self.device)]
        stress_tensor_list = [
            torch.tensor(stress_list, dtype=torch.long).to(self.device)
        ]
        vowel_indices_tensor_list = [
            torch.tensor(vowel_indices, dtype=torch.long).to(self.device)
        ]

        noise_f0_list = [torch.randn(len(phonemes), device=self.device)]
        noise_vuv_list = [torch.randn(len(phonemes), device=self.device)]

        f0_output = self.yukarin_esad_generator(
            noise_f0_list=noise_f0_list,
            noise_vuv_list=noise_vuv_list,
            phoneme_ids_list=phoneme_ids_tensor_list,
            phoneme_durations_list=durations_tensor_list,
            phoneme_stress_list=stress_tensor_list,
            vowel_index_list=vowel_indices_tensor_list,
            speaker_id=numpy.array([f0_speaker_id]),
            step_num=diffusion_step_num,
        )

        f0_vowels = f0_output.f0[0].cpu().numpy()
        vuv_vowels = f0_output.vuv[0].cpu().numpy()

        f0_vowels = f0_vowels + f0_correct
        f0_vowels[vuv_vowels < 0.5] = 0

        rate = self.sampling_rate / self.hop_size

        phoneme_times = numpy.cumsum(numpy.concatenate([[0], durations]))
        frame_length = int(phoneme_times[-1] * rate)

        phoneme_frames = numpy.zeros(frame_length, dtype=numpy.int64)
        for i, phoneme_id in enumerate(phoneme_ids):
            start_frame = int(phoneme_times[i] * rate)
            end_frame = int(phoneme_times[i + 1] * rate)
            end_frame = min(end_frame, frame_length)
            phoneme_frames[start_frame:end_frame] = phoneme_id

        f0_frames = numpy.zeros(frame_length, dtype=numpy.float32)

        vowel_start_times = phoneme_times[vowel_indices]
        vowel_end_times = phoneme_times[numpy.array(vowel_indices) + 1]
        vowel_start_frames = (vowel_start_times * rate).astype(int)
        vowel_end_frames = (vowel_end_times * rate).astype(int)

        for i, (start_frame, end_frame) in enumerate(
            zip(vowel_start_frames, vowel_end_frames, strict=True)
        ):
            f0_frames[start_frame:end_frame] = f0_vowels[i]

        spec_output_size = self.yukarin_esosoad_generator.config.network.output_size
        noise_spec = torch.randn(1, frame_length, spec_output_size, device=self.device)

        spec_output = self.yukarin_esosoad_generator(
            f0=torch.from_numpy(f0_frames).unsqueeze(0).to(self.device),
            phoneme=torch.from_numpy(phoneme_frames).unsqueeze(0).to(self.device),
            noise_spec=noise_spec,
            speaker_id=numpy.array([speaker_id]),
            length=numpy.array([frame_length]),
            step_num=diffusion_step_num,
        )

        spec = spec_output.spec[0]  # (L, C)
        spec = spec.transpose(0, 1).unsqueeze(0).float()  # (1, C, L)

        wave = self.hifigan_generator(spec)
        wave = wave.squeeze().cpu().numpy()

        return wave, (durations, f0_vowels, phoneme_frames, f0_frames, spec)
