import marimo

__generated_with = "0.10.14"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import base64
    import io

    import numpy
    import soundfile
    import torch
    from upath import UPath

    from src.yukarin_esoso_connector.forwarder import (
        ARPA_PHONEMES,
        VOWEL_PHONEMES,
        Forwarder,
        text_to_word_phoneme_data,
    )
    return (
        ARPA_PHONEMES,
        UPath,
        VOWEL_PHONEMES,
        base64,
        io,
        numpy,
        soundfile,
        text_to_word_phoneme_data,
        torch,
        Forwarder,
    )


@app.cell
def __(mo):
    mo.md("# English TTS with Prosody Control")
    return


@app.cell
def __(mo):
    mo.md("## Model Configuration")
    return


@app.cell
def __(mo):
    model_dir_input = mo.ui.text(
        value="./hiho_models/yukarin_es/grad_acc-lr1e-02-d1e-03-ga4-try0",
        label="Yukarin ES Model Directory",
    )
    model_dir_esa_input = mo.ui.text(
        value="./hiho_models/yukarin_esa/init_l1-lr3e-03-d3e-03-try0",
        label="Yukarin ESA Model Directory",
    )
    model_dir_esosoav_input = mo.ui.text(
        value="./hiho_models/yukarin_esosoav/noac-lr2e-04-d1e-02-bs32-try0",
        label="Yukarin ESOSOAV Model Directory",
    )
    speaker_id_input = mo.ui.number(
        value=128,
        label="Speaker ID",
        start=0,
        step=1,
    )
    use_gpu_input = mo.ui.checkbox(
        value=False,
        label="Use GPU",
    )

    mo.vstack([model_dir_input, model_dir_esa_input, model_dir_esosoav_input, speaker_id_input, use_gpu_input])
    return model_dir_input, model_dir_esa_input, model_dir_esosoav_input, speaker_id_input, use_gpu_input


@app.cell
def __(Forwarder, UPath, model_dir_input, model_dir_esa_input, model_dir_esosoav_input, torch, use_gpu_input):
    torch.set_grad_enabled(False)

    forwarder = Forwarder(
        yukarin_es_model_dir=UPath(model_dir_input.value),
        yukarin_esa_model_dir=UPath(model_dir_esa_input.value),
        yukarin_esosoav_model_dir=UPath(model_dir_esosoav_input.value),
        use_gpu=use_gpu_input.value,
    )
    return (forwarder,)


@app.cell
def __(mo):
    mo.md("## Text Input")
    return


@app.cell
def __(mo):
    text_input = mo.ui.text_area(
        value="VOICEVOX is a free text to speech software by Hiho, that utilizes a deep learning.",
        label="Text to synthesize",
    )
    text_input
    return (text_input,)


@app.cell
def __(mo):
    mo.md("## Word and Phoneme Analysis")
    return


@app.cell
def __(text_input, text_to_word_phoneme_data):
    text = text_input.value
    words, phonemes, stress_list, vowel_indices, word_boundaries = text_to_word_phoneme_data(text)
    return phonemes, stress_list, text, vowel_indices, word_boundaries, words


@app.cell
def __(mo, phonemes, stress_list, word_boundaries, words):
    word_phoneme_display = []
    for _i in range(len(word_boundaries) - 1):
        _start_idx = word_boundaries[_i]
        _end_idx = word_boundaries[_i + 1]
        _word_phonemes = phonemes[_start_idx:_end_idx]
        _word_stress = stress_list[_start_idx:_end_idx]
        _word_label = words[_i] if _i < len(words) else "pau"
        word_phoneme_display.append(
            f"**{_word_label}**: {' '.join(_word_phonemes)} (stress: {' '.join(map(str, _word_stress))})"
        )

    word_phoneme_md = mo.md("\n\n".join(word_phoneme_display))
    word_phoneme_md
    return (word_phoneme_display, word_phoneme_md)


@app.cell
def __(mo):
    mo.md("## Stress Control")
    return


@app.cell
def __(mo):
    mo.md("**Note**: 0 = no stress, 1 = primary stress, 2 = secondary stress. Only vowels can have stress.")
    return


@app.cell
def __(mo, phonemes, stress_list, vowel_indices):
    stress_sliders_array = mo.ui.array([
        mo.ui.slider(
            start=0,
            stop=2,
            step=1,
            value=int(stress_list[vowel_indices[_idx]]),
            orientation="vertical",
            label=f"{phonemes[vowel_indices[_idx]]}",
        )
        for _idx in range(len(vowel_indices))
    ])
    return (stress_sliders_array,)


@app.cell
def __(VOWEL_PHONEMES, mo, phonemes, stress_sliders_array, vowel_indices):
    stress_slider_displays = []
    _slider_idx = 0
    for _phoneme_idx in range(len(phonemes)):
        if phonemes[_phoneme_idx] in VOWEL_PHONEMES:
            stress_slider_displays.append(
                mo.vstack([
                    mo.md(f"**{phonemes[_phoneme_idx]}**<br/>{_phoneme_idx}"),
                    stress_sliders_array[_slider_idx],
                ])
            )
            _slider_idx += 1
        else:
            stress_slider_displays.append(
                mo.vstack([
                    mo.md(f"**{phonemes[_phoneme_idx]}**<br/>{_phoneme_idx}"),
                    mo.md("—"),
                ])
            )

    mo.Html(f"<div style='overflow-x: auto; white-space: nowrap;'>{mo.hstack(stress_slider_displays, justify='start', wrap=False)}</div>")
    return (stress_slider_displays,)


@app.cell
def __(stress_list, stress_sliders_array, vowel_indices):
    adjusted_stress_list = stress_list.copy()
    for _idx in range(len(vowel_indices)):
        adjusted_stress_list[vowel_indices[_idx]] = int(stress_sliders_array.value[_idx])

    return (adjusted_stress_list,)


@app.cell
def __(mo):
    mo.md("## Stress Validation")
    return


@app.cell
def __(VOWEL_PHONEMES, adjusted_stress_list, phonemes, word_boundaries, words):
    stress_errors = []

    for _word_idx in range(len(word_boundaries) - 1):
        _start_idx = word_boundaries[_word_idx]
        _end_idx = word_boundaries[_word_idx + 1]
        _word_label = words[_word_idx] if _word_idx < len(words) else "pau"

        _word_vowel_indices = [
            _i for _i in range(_start_idx, _end_idx)
            if phonemes[_i] in VOWEL_PHONEMES
        ]

        if not _word_vowel_indices:
            continue

        _word_stress_values = [adjusted_stress_list[_i] for _i in _word_vowel_indices]

        _stress_1_count = _word_stress_values.count(1)
        _stress_2_count = _word_stress_values.count(2)

        if _stress_1_count != 1:
            stress_errors.append(
                f"Word {_word_idx} ({_word_label}): must have exactly 1 primary stress (1), but has {_stress_1_count}"
            )

        if _stress_2_count > 1:
            stress_errors.append(
                f"Word {_word_idx} ({_word_label}): must have at most 1 secondary stress (2), but has {_stress_2_count}"
            )

    return (stress_errors,)


@app.cell
def __(mo, stress_errors):
    if stress_errors:
        error_msg = "**Stress Validation Errors:**\n\n" + "\n\n".join([f"- {err}" for err in stress_errors])
        mo.callout(error_msg, kind="danger")
    else:
        mo.callout("**Stress Validation:** All rules passed ✓", kind="success")
    return ()


@app.cell
def __(mo):
    show_predictions_toggle = mo.ui.checkbox(value=False, label="Show Predicted Values (Duration and F0)")
    show_predictions_toggle
    return (show_predictions_toggle,)


@app.cell
def __(mo, show_predictions_toggle):
    if show_predictions_toggle.value:
        mo.md("### Predicted Duration (seconds)")
    return ()


@app.cell
def __(ARPA_PHONEMES, forwarder, numpy, phonemes, speaker_id_input, torch):
    phoneme_to_id = {ph: i for i, ph in enumerate(ARPA_PHONEMES)}
    phoneme_ids = numpy.array(
        [phoneme_to_id[ph] for ph in phonemes], dtype=numpy.int64
    )
    phoneme_ids_tensor = torch.from_numpy(phoneme_ids).to(forwarder.device)

    duration_output = forwarder.yukarin_es_generator(
        phoneme_id_list=[phoneme_ids_tensor],
        speaker_id=numpy.array([speaker_id_input.value]),
    )
    predicted_durations = duration_output.duration[0].cpu().numpy()

    predicted_durations[0] = 0.08
    predicted_durations[-1] = 0.08
    predicted_durations = numpy.clip(predicted_durations, 0.01, None)

    return (
        duration_output,
        phoneme_ids,
        phoneme_ids_tensor,
        phoneme_to_id,
        predicted_durations,
    )


@app.cell
def __(mo, phonemes, predicted_durations, show_predictions_toggle):
    if show_predictions_toggle.value:
        duration_text = " | ".join([f"{phonemes[i]}: {predicted_durations[i]:.3f}s" for i in range(len(phonemes))])
        mo.md(f"**Predicted durations**: {duration_text}")
    return ()


@app.cell
def __(mo, show_predictions_toggle):
    if show_predictions_toggle.value:
        mo.md("### Predicted F0 (Pitch) - Log Scale")
    return ()


@app.cell
def __(
    adjusted_stress_list,
    forwarder,
    numpy,
    phoneme_ids_tensor,
    predicted_durations,
    speaker_id_input,
    torch,
    vowel_indices,
):
    phoneme_ids_tensor_list = [phoneme_ids_tensor]
    durations_tensor_list = [
        torch.from_numpy(predicted_durations).float().to(forwarder.device)
    ]
    stress_tensor_list = [torch.tensor(adjusted_stress_list, dtype=torch.long).to(forwarder.device)]
    vowel_indices_tensor_list = [
        torch.tensor(vowel_indices, dtype=torch.long).to(forwarder.device)
    ]

    f0_output = forwarder.yukarin_esa_generator(
        phoneme_ids_list=phoneme_ids_tensor_list,
        phoneme_durations_list=durations_tensor_list,
        phoneme_stress_list=stress_tensor_list,
        vowel_index_list=vowel_indices_tensor_list,
        speaker_id=numpy.array([speaker_id_input.value]),
    )

    predicted_f0_vowels_log = f0_output.f0[0].cpu().numpy()
    predicted_vuv_vowels = f0_output.vuv[0].cpu().numpy()

    return (
        durations_tensor_list,
        f0_output,
        phoneme_ids_tensor_list,
        predicted_f0_vowels_log,
        predicted_vuv_vowels,
        stress_tensor_list,
        vowel_indices_tensor_list,
    )


@app.cell
def __(mo, numpy, phonemes, predicted_f0_vowels_log, show_predictions_toggle, vowel_indices):
    if show_predictions_toggle.value:
        f0_text = " | ".join([f"{phonemes[vowel_indices[i]]}: {numpy.exp(predicted_f0_vowels_log[i]):.1f}Hz (log={predicted_f0_vowels_log[i]:.2f})" for i in range(len(vowel_indices))])
        mo.md(f"**Predicted F0 values**: {f0_text}")
    return ()


@app.cell
def __(mo):
    mo.md("## Prosody Control")
    return


@app.cell
def __(mo):
    mo.md("### Duration Control (seconds)")
    return


@app.cell
def __(mo, phonemes, predicted_durations):
    if predicted_durations is not None:
        _default_durations = predicted_durations
    else:
        _default_durations = [0.08 if phonemes[_i] == "pau" else 0.1 for _i in range(len(phonemes))]
        _default_durations[0] = 0.08
        _default_durations[-1] = 0.08

    duration_slider_widgets = mo.ui.array([
        mo.ui.slider(
            start=0.0,
            stop=1.0 if phonemes[_i] == "pau" else 0.3,
            step=0.01,
            value=float(_default_durations[_i]),
            orientation="vertical",
            label=phonemes[_i],
        )
        for _i in range(len(phonemes))
    ])
    return (duration_slider_widgets,)


@app.cell
def __(duration_slider_widgets, mo, phonemes):
    duration_slider_displays = [
        mo.vstack([
            mo.md(f"**{phonemes[_i]}**"),
            duration_slider_widgets[_i],
            mo.md(f"{duration_slider_widgets.value[_i]:.2f}s"),
        ])
        for _i in range(len(phonemes))
    ]
    mo.Html(f"<div style='overflow-x: auto; white-space: nowrap;'>{mo.hstack(duration_slider_displays, justify='start', wrap=False)}</div>")
    return (duration_slider_displays,)


@app.cell
def __(duration_slider_widgets, numpy):
    durations = numpy.array(duration_slider_widgets.value, dtype=numpy.float32)
    return (durations,)


@app.cell
def __(mo):
    mo.md("### F0 (Pitch) Control for Vowels")
    return


@app.cell
def __(mo):
    mo.md("**Note**: Values are in log scale internally. Displayed values are in Hz (exponential of log values). Range: 3.0-6.5 (log) ≈ 20-665 Hz")
    return


@app.cell
def __(mo, phonemes, predicted_f0_vowels_log, vowel_indices):
    if predicted_f0_vowels_log is not None:
        _default_f0_log = predicted_f0_vowels_log
    else:
        _default_f0_log = [4.5 for _ in range(len(vowel_indices))]

    f0_log_slider_widgets = mo.ui.array([
        mo.ui.slider(
            start=3.0,
            stop=6.5,
            step=0.01,
            value=float(_default_f0_log[_idx]),
            orientation="vertical",
            label=phonemes[vowel_indices[_idx]],
        )
        for _idx in range(len(vowel_indices))
    ])
    return (f0_log_slider_widgets,)


@app.cell
def __(VOWEL_PHONEMES, f0_log_slider_widgets, mo, numpy, phonemes, vowel_indices):
    f0_slider_displays = []
    _slider_idx = 0
    for _phoneme_idx in range(len(phonemes)):
        if phonemes[_phoneme_idx] in VOWEL_PHONEMES:
            _log_value = f0_log_slider_widgets.value[_slider_idx]
            _hz_value = numpy.exp(_log_value)
            f0_slider_displays.append(
                mo.vstack([
                    mo.md(f"**{phonemes[_phoneme_idx]}**"),
                    f0_log_slider_widgets[_slider_idx],
                    mo.md(f"{_hz_value:.1f}Hz"),
                ])
            )
            _slider_idx += 1
        else:
            f0_slider_displays.append(
                mo.vstack([
                    mo.md(f"**{phonemes[_phoneme_idx]}**"),
                    mo.md("—"),
                ])
            )
    mo.Html(f"<div style='overflow-x: auto; white-space: nowrap;'>{mo.hstack(f0_slider_displays, justify='start', wrap=False)}</div>")
    return (f0_slider_displays,)


@app.cell
def __(f0_log_slider_widgets, numpy):
    f0_vowels_log = numpy.array(f0_log_slider_widgets.value, dtype=numpy.float32)
    return (f0_vowels_log,)


@app.cell
def __(mo):
    mo.md("## Audio Synthesis")
    return


@app.cell
def __(
    durations,
    f0_vowels_log,
    forwarder,
    numpy,
    phoneme_ids,
    speaker_id_input,
    vowel_indices,
):
    rate = 24000 / 256

    phoneme_times = numpy.cumsum(numpy.concatenate([[0], durations]))
    frame_length = int(phoneme_times[-1] * rate)

    phoneme_frames = numpy.zeros(frame_length, dtype=numpy.int64)
    for _i, phoneme_id in enumerate(phoneme_ids):
        start_frame = int(phoneme_times[_i] * rate)
        end_frame = int(phoneme_times[_i + 1] * rate)
        end_frame = min(end_frame, frame_length)
        phoneme_frames[start_frame:end_frame] = phoneme_id

    f0_frames_log = numpy.zeros(frame_length, dtype=numpy.float32)

    vowel_start_times = phoneme_times[vowel_indices]
    vowel_end_times = phoneme_times[numpy.array(vowel_indices) + 1]
    vowel_start_frames = (vowel_start_times * rate).astype(int)
    vowel_end_frames = (vowel_end_times * rate).astype(int)

    for _j, (start_frame, end_frame) in enumerate(
        zip(vowel_start_frames, vowel_end_frames, strict=True)
    ):
        f0_frames_log[start_frame:end_frame] = f0_vowels_log[_j]

    wave_outputs = forwarder.yukarin_esosoav_generator(
        f0_list=[f0_frames_log],
        phoneme_list=[phoneme_frames],
        speaker_id=numpy.array([speaker_id_input.value]),
    )

    wave = wave_outputs[0].wave.cpu().numpy()
    return (
        f0_frames_log,
        frame_length,
        phoneme_frames,
        phoneme_times,
        rate,
        vowel_end_frames,
        vowel_end_times,
        vowel_start_frames,
        vowel_start_times,
        wave,
        wave_outputs,
    )


@app.cell
def __(base64, io, mo, soundfile, wave):
    buffer = io.BytesIO()
    soundfile.write(buffer, wave, 24000, format="WAV")
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode()
    audio_html = f'<audio controls src="data:audio/wav;base64,{audio_base64}"></audio>'

    mo.Html(audio_html)
    return audio_base64, audio_html, buffer


if __name__ == "__main__":
    app.run()
