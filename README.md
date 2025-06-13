# Deprecated: Use [HoneyML](https://github.com/hlky/honeyml)

# hlky's AITemplate (HAIT)

HAIT continues development from [facebookincubator/AITemplate](https://github.com/facebookincubator/AITemplate).

Major changes include:

- New kernels
- New features such as workspace allocation modes
- Improved performance
- More modeling [examples](examples/)
  - Companion repo [hlky/diffusers_ait](https://github.com/hlky/diffusers_ait) implements models from [Diffusers](https://github.com/huggingface/diffusers)
  - [Transformers](https://github.com/huggingface/transformers) soon:tm:


## Installation

### Requirements

Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

For Windows, install [Build Tools for Visual Studio 2022](https://aka.ms/vs/17/release/vs_BuildTools.exe) or Visual Studio 2022 ([Community](https://aka.ms/vs/17/release/vs_community.exe), [Professional](https://aka.ms/vs/17/release/vs_professional.exe) or [Enterprise](https://aka.ms/vs/17/release/vs_enterprise.exe)).

If Build Tools or Visual Studio is installed after CUDA Toolkit, re-run CUDA Toolkit installation to get Visual Studio integration. Build Tools only may need `CUDA X.Y.props` copying manually, refer to [CUDA documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html?highlight=visual%20studio#sample-projects) for the default `.props` install directory and adjust for your Build Tools install directory.


### Python
```
git clone --recursive https://github.com/hlky/AITemplate
cd AITemplate/python
pip install -e .
```

### Build release

```
python setup.py bdist_wheel
pip uninstall -y aitemplate
pip install dist/*.whl
```

## Roadmap

Generally:

- More kernels
- More modeling support

Refer to [Issues](https://github.com/hlky/AITemplate/issues) and [Projects](https://github.com/hlky/AITemplate/projects).

## Acknowledgements

With thanks to the original developers and other Meta engineers.

## License

AITemplate is licensed under the [Apache 2.0 License](LICENSE).
