import base64
import os
from typing import List

import replicate

import requests


def download_file(url, local_filename):
    """
    Download a file from a URL and save it to a local file.

    Args:
    - url (str): URL of the file to download.
    - local_filename (str): Path to save the file to.

    Returns:
    - str: Path to the downloaded file.
    """
    # Send a GET request to the URL
    with requests.get(url, stream=True) as r:
        # Raise an exception for bad responses
        r.raise_for_status()
        # Open the local file for writing in binary mode
        with open(local_filename, 'wb') as f:
            # Write the content to the local file in chunks
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def to_base64(input_path: str) -> str:
    with open(input_path, 'rb') as file:
        data = base64.b64encode(file.read()).decode('utf-8')
        input = f"data:application/octet-stream;base64,{data}"
    return input


def audio2vector(input_path: str) -> List[float]:
    # output dim: 1024
    input_b64 = to_base64(input_path)
    output = replicate.run(
        "daanelson/imagebind:0383f62e173dc821ec52663ed22a076d9c970549c209666ac3db181618b7a304",
        input={
            "input": input_b64,
            "modality": "audio",
            "text_input": ""
        }
    )
    print(output)
    return output


def voice_cloning(input_path_audio_src: str, prompt: str):
    input_b64 = to_base64(input_path_audio_src)
    output = replicate.run(
        "cjwbw/openvoice:af9877f21c4e040357eb6424ecddd7199367be2d8667ad4b6bbd306cbcd326e4",
        input={
            "audio": input_b64,
            "speed": 1,
            "style": "default",
            "prompt": prompt,
            "agree_terms": True
        }
    )
    print(output)
    return output


if __name__ == "__main__":
    input_path = "audio_samples/wav/audio_2024-03-11_10-57-30.wav"
    input_path2clone = "audio_samples/samples_converted/LJ037-0171.wav"

    result = audio2vector(input_path=input_path)

    prompt = "Tom has a small dog named Max. Max is black and white and very playful."
    result_vc = voice_cloning(input_path_audio_src=input_path2clone, prompt=prompt)
    download_file(result_vc, local_filename=os.path.join("audio_samples/voice_cloned",
                                                         f"{prompt.replace(' ', '_')}_{os.path.basename(input_path2clone)}"))

    prompt = "Every morning, Tom and Max go for a walk in the park."
    result_vc_2 = voice_cloning(input_path_audio_src=input_path2clone, prompt=prompt)
    download_file(result_vc_2,
                  local_filename=os.path.join("audio_samples/voice_cloned",
                                              f"{prompt.replace(' ', '_')}_{os.path.basename(input_path2clone)}"))
    h = None
