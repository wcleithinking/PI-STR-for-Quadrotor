# Licensing status

> **This document is not legal advice.** It records what is in the repository, what
> the current copyright status is, and what someone with authority has to do before
> this code can be redistributed or reused. If money, a company, or a publication is
> involved, talk to a lawyer or your organisation's open-source office.

## TL;DR

**This repository has no license.** The upstream repository it comes from
(`wcleithinking/PI-STR-for-Quadrotor`) never had one either. Absent an explicit
license, the default is *all rights reserved*: nobody may copy, modify, or
redistribute this code without the copyright holder's permission. It is **not** open
source, and adding an MIT/Apache-2.0/BSD license here would be a false claim that
nobody in this repository is entitled to make.

Two concrete consequences:

1. Do not add a `LICENSE` file, an SPDX identifier, or a `license:` field in
   `CITATION.cff` unless and until the copyright holder grants one **in writing**.
2. The vendored `firmware/libraries/SdFat/` is third-party **GPL-3.0-or-later** code
   (verified below). Redistributing this repository, or firmware built from it,
   already raises a GPL obligation that nobody has resolved.

## What is in the tree

| Path | Origin | License status |
|---|---|---|
| `firmware/Quadrotor_Mode_I_Arduino_Due/`, `.../Mode_II.../` | © Wenchao Lei, 2018 | **no license**; headers carry a warranty disclaimer only |
| `firmware/libraries/MyLibraries_Wenchao/` | © Wenchao Lei, 2018 | **no license**, no per-file headers |
| `firmware/libraries/SdFat/` | © William Greiman, 2012 (2015.4.26) | **GPL-3.0-or-later** — per-file headers, but no license text shipped |
| `sim/matlab/` | © Wenchao Lei, 2018 | **no license** |
| `sim/python/`, `tests/`, `docs/`, `Makefile`, `platformio.ini`, `CITATION.cff`, `README.md`, `.gitignore`, `.gitattributes` | added during this refactor | **no license** — same restriction, and they are derivative of the code above |
| The paper (IEEE TCST 27(6), 2019) | © IEEE | not covered by any code license; see §5 |

How this was checked (repeat it yourself, do not take it on trust):

```bash
# 1. Was a LICENSE/COPYING file ever added, in any commit, on any branch?
git log --all --diff-filter=A --name-only --format='%h' | grep -iE 'licen|copying'
#    -> no output

# 2. Is there a license file at the root today?
ls -a | grep -iE 'licen|copying'
#    -> no output

# 3. What do the .ino file headers actually say?
sed -n '1,20p' firmware/Quadrotor_Mode_II_Arduino_Due/Quadrotor_Mode_II_Arduino_Due.ino

# 4. Is the vendored SdFat GPL?
sed -n '1,20p' firmware/libraries/SdFat/src/SdFat.cpp
```

## 1. What "no license" means

Copyright is automatic: the moment code is written, its author holds the exclusive
rights to copy, modify, and distribute it. A license is the *permission* others need
to do those things. Without one, the default is "all rights reserved".

* **Publicly readable is not licensed.** Publishing source on GitHub is not a grant.
  The GitHub Terms of Service give other GitHub users a narrow right to view and to
  fork *inside GitHub* (and to reproduce the content as GitHub's features allow); they
  do not grant a right to reuse the code elsewhere, to redistribute it, or to build a
  product on it.
* **"No license" does not mean "public domain".** Nothing here has been dedicated to
  the public domain.
* **Citing the paper is not a license.** `CITATION.cff` and academic norms tell people
  how to acknowledge the work; they grant no rights.
* **A warranty disclaimer is not a license.** The header of each `.ino` file:

  > The Software is provided "as is", without warranty of any kind, express or
  > implied, including but not limited to the warranties of merchantability, fitness
  > for a particular purpose and noninfringement. In no event shall the authors or
  > copyright holders be liable for any claim, damages or other liability…

  That paragraph limits liability. It does not permit copying or redistribution.
* **Silence from the author is not permission.** Only the copyright holder (or an
  authorised representative) can grant a license, and it should be granted in a form
  you can show later.

## 2. Third-party code: SdFat is GPL-3.0

`firmware/libraries/SdFat/` is a vendored copy of William Greiman's Arduino SdFat
library (`library.properties` says version `2015.4.26`). Every source file carries:

> This Library is free software: you can redistribute it and/or modify it under the
> terms of the GNU General Public License as published by the Free Software
> Foundation, either version 3 of the License, or (at your option) any later version.
> … You should have received a copy of the GNU General Public License along with the
> Arduino SdFat Library.

So the firmware already contains GPL-3.0-or-later code, and **no copy of the GPL text
is shipped** (there is no `COPYING`/`LICENSE` anywhere in the tree). If the firmware
is distributed — as source, or as a compiled `.bin`/`.hex` — the GPL's conditions
apply to the combined work: provide the license text and the corresponding source,
and do not add restrictions that conflict with the GPL.

This is exactly the kind of question that needs the copyright holder, because only
they can decide how the rest of the firmware is licensed and whether the GPL-covered
part is acceptable to them. Do not "solve" it by deleting the SdFat headers.

The Arduino core and the `Servo` library are **not** vendored here; they are build
dependencies provided by the Arduino/PlatformIO toolchain, so they are not
redistributed by this repository — but if you ship a firmware binary, check their
licenses too.

## 3. Risk table

| Situation | Exposure |
|---|---|
| You read the code and re-implement the algorithm from the paper | low, but a faithful line-by-line port may be a derivative work — get advice if it matters |
| You fork on GitHub and open a PR upstream | acceptable in practice (that is what forks are for); do not present the fork as licensed code |
| You redistribute this repository (mirror, vendored copy in another project, Docker image, course material) | **infringement risk** — no permission has been granted |
| You use it in a product, or ship firmware built from it | **highest risk**: redistribution plus the unresolved GPL question |
| You publish results obtained with it | allowed in practice with citation; the *code* stays unlicensed |
| You paste files into a differently-licensed project | contaminates that project; the files do not become MIT/Apache by being copied |
| You add `LICENSE` = MIT without asking | a false grant; you cannot license someone else's code |

## 4. Options, best first

**A. Ask the copyright holder for a license (recommended).** Wenchao Lei is reachable
at the address published in the upstream README (`wcleithinking@gmail.com`) and on
GitHub as [`@wcleithinking`](https://github.com/wcleithinking). Ask for a permissive
license (MIT or Apache-2.0 are the usual choices) in writing — an email reply that
says "you may license it under MIT" is normally enough for an academic project, a
signed CLA is better if a company is involved. Apache-2.0 additionally gives an
explicit patent grant, which matters for control/firmware work.

**B. Keep it unlicensed and link to upstream.** Use it locally, point others at
`https://github.com/wcleithinking/PI-STR-for-Quadrotor`, and never redistribute. Zero
legal risk, but the repository cannot be used by anyone else and cannot be
republished.

**C. Contribute the improvements back upstream.** If the refactoring is offered as a
pull request to the original repository, the author's own repository remains the
single distribution point and no new license question is created. This is the
lowest-friction path for the new `sim/python/` and `docs/` work.

**D. Clean-room rewrite from the paper.** A genuinely independent implementation
(taken from the paper, not from this source) is the only way to get an unencumbered
codebase. It is expensive and easy to get wrong; keep the original sources out of the
room where the rewrite happens, and document the process if it matters.

## 5. The paper is separate from the code

The article is © IEEE and published under IEEE's copyright terms. That covers the
text and figures, not this repository's source files, and it grants no rights to the
code. Likewise, the DOI in `CITATION.cff` is metadata: citing the paper is correct
academic practice and is what the authors ask for, but it is not a license to reuse
the code. If you reuse figures from the paper in derived work, that is an IEEE
permission question, not a GitHub one.

## 6. Recommended actions, in order

1. **Do not add a license file.** If someone already did, revert it and record why in
   this document.
2. **Open an issue / send an email to the author** asking for a license. Keep the
   reply. Suggested text:

   > Subject: License for PI-STR-for-Quadrotor
   >
   > Dear Dr. Lei,
   >
   > I am working on a refactor of the code in
   > https://github.com/wcleithinking/PI-STR-for-Quadrotor (the TCST 2019 paper
   > "Robust Adaptive Tracking Control for Quadrotors by Combining PI and
   > Self-Tuning Regulator"). The repository has no LICENSE file, so as it stands
   > nobody may reuse or redistribute it.
   >
   > Would you be willing to release the code (the firmware, the MATLAB study, and
   > any later contributions you approve) under a permissive license such as MIT or
   > Apache-2.0? A short written confirmation is enough for us; we will add the
   > LICENSE file and credit you and the paper.
   >
   > Thank you — [name], [affiliation]

3. **Note the SdFat GPL question in the same conversation.** Ask whether the author
   is aware that `firmware/libraries/SdFat/` is GPL-3.0-or-later and how they want it
   handled (e.g. drop the vendored copy and use the platform's package, or ship the
   GPL text and source as the GPL requires).
4. **Until then, treat the repository as internal.** No mirror, no packaged release,
   no vendoring into other projects, no bundled `LICENSE`.
5. **Make the status visible** (already done): the root `README.md` says the code is
   unlicensed and points here, and `CITATION.cff` deliberately omits `license:`.

## 7. When permission is granted

Do it in one dedicated commit so the grant is auditable:

- [ ] add `LICENSE` with the granted text (and `NOTICE` if Apache-2.0);
- [ ] add `license: MIT` (or the granted identifier) to `CITATION.cff`;
- [ ] update the "License" section of `README.md` and remove the warning box;
- [ ] add a short SPDX header to new source files, e.g. `// SPDX-License-Identifier: MIT`;
- [ ] for SdFat: either ship `COPYING` + written offer/source as GPL-3.0 requires, or
      remove the vendored copy and depend on the platform package — record the choice;
- [ ] append to this document: what was granted, by whom, when, and where the
      evidence is stored (link the email/issue).

## 8. Do not

- do not add `MIT`, `Apache-2.0`, `GPL-3.0`, or any other license header to files you
  did not author;
- do not describe the project as "open source" in the README, a paper, or a release
  note while no license exists;
- do not publish a package on PyPI/npm with this code inside;
- do not delete third-party copyright headers (SdFat) to make the question go away;
- do not assume "it was on GitHub, so it is free".
