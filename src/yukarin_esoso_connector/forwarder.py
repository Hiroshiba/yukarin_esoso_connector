from pathlib import Path

import numpy
import torch
import yaml
from g2p_en import G2p
from upath import UPath
from yukarin_es.config import Config as ConfigEs
from yukarin_es.generator import Generator as GeneratorEs
from yukarin_es.utility.upath_utility import to_local_path
from yukarin_esa.config import Config as ConfigEsa
from yukarin_esa.generator import Generator as GeneratorEsa
from yukarin_esosoav.config import Config as ConfigEsosoav
from yukarin_esosoav.generator import Generator as GeneratorEsosoav

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


class Forwarder:
    def __init__(
        self,
        yukarin_es_model_dir: UPath | Path,
        yukarin_esa_model_dir: UPath | Path,
        yukarin_esosoav_model_dir: UPath | Path,
        use_gpu: bool,
        yukarin_es_iteration: int | None = None,
        yukarin_esa_iteration: int | None = None,
        yukarin_esosoav_iteration: int | None = None,
    ):
        yukarin_es_model_dir = UPath(yukarin_es_model_dir)
        yukarin_esa_model_dir = UPath(yukarin_esa_model_dir)
        yukarin_esosoav_model_dir = UPath(yukarin_esosoav_model_dir)

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

        config_esa = ConfigEsa.from_dict(
            yaml.safe_load((yukarin_esa_model_dir / "config.yaml").read_text())
        )

        predictor_esa_path = get_predictor_model_path(
            yukarin_esa_model_dir, iteration=yukarin_esa_iteration
        )
        print("yukarin_esa predictor:", predictor_esa_path)
        self.yukarin_esa_generator = GeneratorEsa(
            config=config_esa,
            predictor=to_local_path(predictor_esa_path),
            use_gpu=use_gpu,
        )
        self.yukarin_esa_generator.predictor.apply(remove_weight_norm)
        print("yukarin_esa loaded!")

        config_esosoav = ConfigEsosoav.from_dict(
            yaml.safe_load((yukarin_esosoav_model_dir / "config.yaml").read_text())
        )

        predictor_esosoav_path = get_predictor_model_path(
            yukarin_esosoav_model_dir, iteration=yukarin_esosoav_iteration
        )
        print("yukarin_esosoav predictor:", predictor_esosoav_path)
        self.yukarin_esosoav_generator = GeneratorEsosoav(
            config=config_esosoav,
            predictor=to_local_path(predictor_esosoav_path),
            use_gpu=use_gpu,
        )
        self.yukarin_esosoav_generator.predictor.apply(remove_weight_norm)
        print("yukarin_esosoav loaded!")

        self.device = self.yukarin_es_generator.device
        self.phoneme_to_id = {ph: i for i, ph in enumerate(ARPA_PHONEMES)}

    @torch.no_grad()
    def forward(
        self,
        text: str,
        speaker_id: int,
        f0_speaker_id: int | None = None,
        f0_correct: float = 0,
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

        f0_output = self.yukarin_esa_generator(
            phoneme_ids_list=phoneme_ids_tensor_list,
            phoneme_durations_list=durations_tensor_list,
            phoneme_stress_list=stress_tensor_list,
            vowel_index_list=vowel_indices_tensor_list,
            speaker_id=numpy.array([f0_speaker_id]),
        )

        f0_vowels = f0_output.f0[0].cpu().numpy()
        vuv_vowels = f0_output.vuv[0].cpu().numpy()

        f0_vowels = f0_vowels + f0_correct
        f0_vowels[vuv_vowels < 0.5] = 0

        rate = 24000 / 256

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
            zip(vowel_start_frames, vowel_end_frames)
        ):
            f0_frames[start_frame:end_frame] = f0_vowels[i]

        wave_outputs = self.yukarin_esosoav_generator(
            f0_list=[f0_frames],
            phoneme_list=[phoneme_frames],
            speaker_id=numpy.array([speaker_id]),
        )

        wave = wave_outputs[0].wave.cpu().numpy()

        return wave, (durations, f0_vowels, phoneme_frames, f0_frames)
