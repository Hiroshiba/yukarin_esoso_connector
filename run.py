import argparse
from pathlib import Path

import soundfile
from upath import UPath

from yukarin_esoso_connector.forwarder import Forwarder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yukarin_es_model_dir", required=True, type=UPath)
    parser.add_argument("--yukarin_esa_model_dir", required=True, type=UPath)
    parser.add_argument("--yukarin_esosoav_model_dir", required=True, type=UPath)
    parser.add_argument("--yukarin_es_iteration", type=int)
    parser.add_argument("--yukarin_esa_iteration", type=int)
    parser.add_argument("--yukarin_esosoav_iteration", type=int)
    parser.add_argument("--output_dir", type=Path, default=Path("./output"))
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--text", type=str, nargs="+")
    parser.add_argument("--text_path", type=Path)
    parser.add_argument("--speaker_id", type=int, nargs="+", required=True)
    parser.add_argument("--f0_speaker_id", type=int)
    parser.add_argument("--f0_correct", type=float, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    forwarder = Forwarder(
        yukarin_es_model_dir=args.yukarin_es_model_dir,
        yukarin_esa_model_dir=args.yukarin_esa_model_dir,
        yukarin_esosoav_model_dir=args.yukarin_esosoav_model_dir,
        use_gpu=args.use_gpu,
        yukarin_es_iteration=args.yukarin_es_iteration,
        yukarin_esa_iteration=args.yukarin_esa_iteration,
        yukarin_esosoav_iteration=args.yukarin_esosoav_iteration,
    )

    if args.text:
        texts = args.text
    elif args.text_path:
        texts = args.text_path.read_text().strip().split("\n")
    else:
        raise ValueError("Either --text or --text_path must be specified")

    speaker_ids = args.speaker_id

    total = len(texts) * len(speaker_ids)
    count = 0

    for text_idx, text in enumerate(texts):
        for speaker_id in speaker_ids:
            count += 1
            print(
                f"\n[{count}/{total}] Synthesizing: {text} (text_idx={text_idx}, speaker_id={speaker_id})"
            )

            wave, _ = forwarder.forward(
                text=text,
                speaker_id=speaker_id,
                f0_speaker_id=args.f0_speaker_id,
                f0_correct=args.f0_correct,
            )

            output_path = (
                args.output_dir / f"output_text{text_idx}_speaker{speaker_id}.wav"
            )
            soundfile.write(output_path, wave, 24000)
            print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
