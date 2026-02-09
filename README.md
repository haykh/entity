# `Entity` 

tl;dr: One particle-in-cell code to rule them all. 

`Entity` is a community-driven open-source coordinate-agnostic general-relativistic (GR) particle-in-cell (PIC) code written in C++17 specifically targeted to study plasma physics in relativistic astrophysical systems. The main algorithms of the code are written in covariant form, allowing to easily implement arbitrary grid geometries. The code is highly modular, and is written in the architecture-agnostic way using the [`Kokkos`](https://kokkos.org/kokkos-core-wiki/) performance portability library, allowing the code to efficiently use device parallelization on CPU and GPU architectures of different types. The multi-node parallelization is implemented using the `MPI` library, and the data output is done via the [`ADIOS2`](https://github.com/ornladios/ADIOS2) library which supports multiple output formats, including `HDF5` and `BP5`.

`Entity` is part of the `Entity toolkit` framework, which also includes a Python library for fast and efficient data analysis and visualization of the simulation data: [`nt2py`](https://pypi.org/project/nt2py/).

Our [detailed documentation](https://entity-toolkit.github.io/) includes everything you need to know to get started with using and/or contributing to the `Entity toolkit`. If you find bugs or issues, please feel free to add a GitHub issue or submit a pull request. Users with significant contributions to the code will be added to the list of developers, and assigned an emoji of their choice (important!).

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## News

- [Dec 2025]: **high-order** shape functions [PR #109](https://github.com/entity-toolkit/entity/pull/109) and advanced field stencils [PR #103](https://github.com/entity-toolkit/entity/pull/103) are now supported.
- [Dec 2025]: **particle tracking** is now fully supported via [PR #144](https://github.com/entity-toolkit/entity/pull/144).
- [Nov 2025]: our **method papers** are online: [Special relativistic module](https://ui.adsabs.harvard.edu/abs/2025arXiv251117710H/abstract), [GR module](https://ui.adsabs.harvard.edu/abs/2025arXiv251117701G/abstract)!

## Citation

Please, see the `CITATION` document for the relevant BibTeX entries if you would like to cite one of the method papers for the code.

## Join the community

Everyone is welcome to join our small yet steadily growing community of code users and developers; regardless of how much you are planning to contribute -- we always welcome fresh ideas and feedback. We hold weekly Slack calls on Mondays at 12pm NY time, and have a dedicated Slack channel where you can be easily added by emailing one of the maintainers (indicated with an asterisk in the list below). Anyone is welcome to join both our **Slack workspace** and the weekly meetings -- please feel free to request access by emailing.

Another way of contacting us is via GitHub issues and/or pull requests. Make sure to check out our [F.A.Q.](https://entity-toolkit.github.io/wiki/content/1-getting-started/9-faq/), as it might help you answer your question.

> Keep in mind, you are free to use the code in any capacity, and there is absolutely no requirement on our end of including any of the developers in your project/proposal (as highlighted in our Code of Conduct). When contributing, also keep in mind that the code you upload to the repository automatically becomes public and open-source, and the same standards will be applied to it as to the rest of the code. 

## Contributors (alphabetical)

Maintainers indicated with an arrow.

* :guitar: Ludwig Böss {[@LudwigBoess](https://github.com/LudwigBoess)}
* :eyes: Yangyang Cai {[@StaticObserver](https://github.com/StaticObserver)}
* :tipping_hand_person: Alexander Chernoglazov {[@SChernoglazov](https://github.com/SChernoglazov)}
* :tea: Benjamin Crinquand {[@bcrinquand](https://github.com/bcrinquand)}
* :bubble_tea: Alisa Galishnikova {[@alisagk](https://github.com/alisagk)}
* :steam_locomotive: Evgeny Gorbunov {[@Alcauchy](https://github.com/Alcauchy)} [-> [genegorbs [at] gmail](mailto:genegorbs@gmail.com)]
* :coffee: Hayk Hakobyan {[@haykh](https://github.com/haykh)} [-> [haykh.astro [at] gmail](mailto:haykh.astro@gmail.com)]
* :potato: Jens Mahlmann {[@jmahlmann](https://github.com/jmahlmann)}
* :dolphin: Sasha Philippov {[@sashaph](https://github.com/sashaph)}
* :radio: Siddhant Solanki {[@sidruns30](https://github.com/sidruns30)}
* :shrug: Arno Vanthieghem {[@vanthieg](https://github.com/vanthieg)}
* :cat: Muni Zhou {[@munizhou](https://github.com/munizhou)}

## Branch policy

- `master` branch contains the latest stable version of the code which has already been released.
- Development on the core is done on branches starting with `dev/`.
- Bug-fixes are being pushed to branches starting with `bug/`.
- All `bug/` and `dev/` branches must have an open pull-request describing in detail its purpose.
- Before merging to the master branch, all the branches must first be merged to the latest release-candidate branch, which ends with `rc`, via a pull request. This can either be a major release: `1.X.0rc`, or a patch release `1.X.Yrc`. 
- Stale branches will be archived with a tag starting with `archive/` (can still be accessed via the "Tags" tab) and removed.

