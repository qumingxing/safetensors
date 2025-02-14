import os
from typing import Dict, Optional, Union

import numpy as np
import tensorflow as tf

from safetensors import numpy, safe_open


def save(tensors: Dict[str, tf.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, tf.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.tensorflow import save
    import tensorflow as tf

    tensors = {"embedding": tf.zeros((512, 1024)), "attention": tf.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    np_tensors = _tf2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(
    tensors: Dict[str, tf.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Saves a dictionary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, tf.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`, or `os.PathLike`)):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `None`

    Example:

    ```python
    from safetensors.tensorflow import save_file
    import tensorflow as tf

    tensors = {"embedding": tf.zeros((512, 1024)), "attention": tf.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    np_tensors = _tf2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata)


def load(data: bytes) -> Dict[str, tf.Tensor]:
    """
    Loads a safetensors file into tensorflow format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, tf.Tensor]`: dictionary that contains name as key, value as `tf.Tensor` on cpu

    Example:

    ```python
    from safetensors.tensorflow import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = numpy.load(data)
    return _np2tf(flat)


def load_file(filename: Union[str, os.PathLike]) -> Dict[str, tf.Tensor]:
    """
    Loads a safetensors file into tensorflow format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors

    Returns:
        `Dict[str, tf.Tensor]`: dictionary that contains name as key, value as `tf.Tensor`

    Example:

    ```python
    from safetensors.tensorflow import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    file_size = os.path.getsize(filename)
    key = '8cc72b05705d5c46f412af8cbed55aad'  # 32 字节密钥（AES-256）
    iv = '667b02a85c61c786def4521b060265e8'  # 16 字节初始化向量
    cipher = AES.new(bytes.fromhex(key), AES.MODE_CBC, bytes.fromhex(iv))

    with open(filename, "rb") as f:
        byte_array = f.read(8)
        header_len = (byte_array[0] & 0xFF) \
        | ((byte_array[1] & 0xFF) << 8) \
        | ((byte_array[2] & 0xFF) << 16) \
        | ((byte_array[3] & 0xFF) << 24)

        f.seek(8)
        metadata = f.read(header_len)
        json_obj = json.loads(metadata.decode('utf-8'))

        file_data = f.read()
        decrypted_data = byte_array+metadata+unpad(cipher.decrypt(file_data), AES.block_size)

        for key, value in json_obj.items():
            if key == '__metadata__':
                continue

            start, end = value['data_offsets']
            shape = tuple(value['shape']) if len(value['shape']) == 2 else None
            dtype = torch.float32 if value['dtype'] == 'F32' else torch.float64

            offset = header_len + 8 + start
            num_bytes = end - start

            #f.seek(offset)
            #buffer = f.read(num_bytes)
            buffer = decrypted_data[offset : offset + num_bytes]
            tensor = torch.frombuffer(buffer, dtype=dtype)
            if device != "cpu":
                if shape:
                    tensor = tensor.reshape(shape).to(device)
                else:
                    tensor = tensor.to(device)
            else:
                if shape:
                    tensor = tensor.reshape(shape)
            result[key] = tensor

    return result
    """
    with safe_open(filename, framework="tf") as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result
    """

def _np2tf(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = tf.convert_to_tensor(v)
    return numpy_dict


def _tf2np(tf_dict: Dict[str, tf.Tensor]) -> Dict[str, np.array]:
    for k, v in tf_dict.items():
        tf_dict[k] = v.numpy()
    return tf_dict
