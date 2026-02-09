#!/usr/bin/env python3

from __future__ import annotations

import curses
import json
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# ============================
# colors: edit these
# ============================

# foreground colors (use curses.COLOR_* or -1 for default)
COLOR_TITLE_FG = curses.COLOR_BLUE
COLOR_TEXT_FG = curses.COLOR_WHITE
COLOR_SELECTED_FG = curses.COLOR_WHITE
COLOR_SELECTED_BG = curses.COLOR_BLACK
COLOR_HINT_FG = curses.COLOR_YELLOW
COLOR_OK_FG = curses.COLOR_GREEN
COLOR_ERR_FG = curses.COLOR_RED
COLOR_KEY_FG = curses.COLOR_MAGENTA
COLOR_DIM_FG = curses.COLOR_CYAN

# pair IDs (must be unique small ints)
PAIR_TITLE = 1
PAIR_TEXT = 2
PAIR_SELECTED = 3
PAIR_HINT = 4
PAIR_OK = 5
PAIR_ERR = 6
PAIR_KEY = 7
PAIR_DIM = 8


KOKKOS_BACKENDS = ["cpu", "cuda", "hip", "sycl"]
ADIOS2_MPI_MODES = ["non-mpi", "mpi"]

MESSAGE: str = ""


@dataclass
class Settings:
    cluster: str = "(custom)"
    write_modulefiles: bool = False
    overwrite: bool = False
    install_prefix: str = os.path.join(os.path.expanduser("~"), ".entity")

    apps: dict = field(
        default_factory=lambda: {"Kokkos": False, "adios2": False, "nt2py": False}
    )

    # versions
    kokkos_version: str = "5.0.1"
    adios2_version: str = "2.11.0"

    # options
    kokkos_backend: str = "cpu"
    kokkos_arch: str = ""
    extra_kokkos_flags: List[str] = field(default_factory=list)
    adios2_mpi: str = "non-mpi"
    extra_adios2_flags: List[str] = field(default_factory=list)

    module_loads: List[str] = field(default_factory=list)

    def from_json(self, json_str: str) -> None:
        data = json.loads(json_str)
        self.cluster = data.get("cluster", self.cluster)
        self.write_modulefiles = data.get("write_modulefiles", self.write_modulefiles)
        self.overwrite = data.get("overwrite", self.overwrite)
        self.install_prefix = data.get("install_prefix", self.install_prefix)
        self.apps = data.get("dependencies", self.apps)
        versions = data.get("versions", {})
        self.kokkos_version = versions.get("Kokkos", self.kokkos_version)
        self.adios2_version = versions.get("adios2", self.adios2_version)
        options = data.get("options", {})
        self.kokkos_backend = options.get("kokkos_backend", self.kokkos_backend)
        self.kokkos_arch = options.get("kokkos_arch", self.kokkos_arch)
        self.adios2_mpi = options.get("adios2_mpi", self.adios2_mpi)
        self.module_loads = data.get("module_loads", self.module_loads)

    def apps_summary(self) -> str:
        chosen = [k for k, v in self.apps.items() if v]
        return ", ".join(chosen) if chosen else "(none)"

    def to_json(self) -> str:
        return json.dumps(
            {
                "cluster": self.cluster,
                "write_modulefiles": self.write_modulefiles,
                "overwrite": self.overwrite,
                "install_prefix": self.install_prefix,
                "dependencies": self.apps,
                "versions": {
                    "Kokkos": self.kokkos_version,
                    "adios2": self.adios2_version,
                },
                "options": {
                    "kokkos_backend": self.kokkos_backend,
                    "kokkos_arch": self.kokkos_arch,
                    "adios2_mpi": self.adios2_mpi,
                },
                "module_loads": self.module_loads,
            },
            indent=2,
        )


def unindent(script: str) -> str:
    script_lines = script.splitlines()
    min_indent = min(
        (len(line) - len(line.lstrip()) for line in script_lines if line.strip()),
        default=0,
    )
    trimmed_lines = [line[min_indent:] for line in script_lines]
    if trimmed_lines[0] == "":
        trimmed_lines = trimmed_lines[1:]
    if trimmed_lines[-1] == "":
        trimmed_lines = trimmed_lines[:-1]
    return "\n".join(trimmed_lines)


def InstallKokkosScriptModfile(settings: Settings) -> tuple[str, str]:
    if settings.apps.get("Kokkos", False):
        prefix = settings.install_prefix
        version = settings.kokkos_version
        backend = settings.kokkos_backend
        arch = settings.kokkos_arch.strip()
        modules = "\n".join(
            [f"module load {module} && \\" for module in settings.module_loads]
        )
        src_path = f"{prefix}/src/kokkos"
        install_path = (
            f"{prefix}/kokkos/{version}/{backend}{f'_{arch}' if arch else ''}"
        )
        if os.path.exists(install_path) and not settings.overwrite:
            raise FileExistsError(
                f"Kokkos install path {install_path} already exists and overwrite is disabled"
            )

        extra_flags = "-D ".join(settings.extra_kokkos_flags)
        cxx_standard = 20 if tuple(map(int, version.split("."))) >= (5, 0, 0) else 17

        if arch == "":
            arch = "NATIVE"
        arch = arch.upper()

        script = f"""
        # Kokkos installation
        {modules}
        rm -rf {src_path} && \\
        git clone https://github.com/kokkos/kokkos.git {src_path} && \\
        cd {src_path} && \\
        git checkout {version} && \\
        cmake -B build \\
            -D CMAKE_CXX_STANDARD={cxx_standard} \\
            -D CMAKE_CXX_EXTENSIONS=OFF \\
            -D CMAKE_POSITION_INDEPENDENT_CODE=TRUE \\
            -D Kokkos_ARCH_{arch}=ON {f'-D Kokkos_ENABLE_{backend.upper()}=ON' if backend != 'cpu' else ''} \\
            -D CMAKE_INSTALL_PREFIX={install_path} {extra_flags} && \\
        cmake --build build -j $(nproc) && \\
        cmake --install build
        """

        modfile = f"""
        #%Module1.0######################################################################
        ##
        ## Kokkos @ {backend} @ {arch} modulefile
        ##
        #################################################################################
        proc ModulesHelp {{ }} {{
            puts stderr \"\\tKokkos @ {backend} @ {arch}\\n\"
        }}

        module-whatis      \"Sets up Kokkos @ {backend} @ {arch}\"

        conflict           kokkos
        {modules}

        set                basedir      {install_path}
        prepend-path       PATH         $basedir/bin
        setenv             Kokkos_DIR   $basedir

        setenv Kokkos_ARCH_{arch} ON
        {f'setenv Kokkos_ENABLE_{backend.upper()} ON' if backend != 'cpu' else ''}
        """

        return (unindent(script), unindent(modfile))

    else:
        return ("""# skipping Kokkos install""", "")


def InstallAdios2Script(settings: Settings) -> tuple[str, str]:
    if settings.apps.get("adios2", False):
        prefix = settings.install_prefix
        version = settings.adios2_version
        mpi_mode = settings.adios2_mpi
        modules = "\n".join(
            [f"module load {module} && \\" for module in settings.module_loads]
        )
        src_path = f"{prefix}/src/adios2"
        install_path = f"{prefix}/adios2/{version}/{mpi_mode}"
        if os.path.exists(install_path) and not settings.overwrite:
            raise FileExistsError(
                f"Adios2 install path {install_path} already exists and overwrite is disabled"
            )

        extra_flags = "-D ".join(settings.extra_adios2_flags)
        cxx_standard = (
            20
            if tuple(map(int, settings.kokkos_version.split("."))) >= (5, 0, 0)
            else 17
        )

        with_mpi = "ON" if mpi_mode == "mpi" else "OFF"

        script = f"""
        # Adios2 installation
        {modules}
        rm -rf {src_path} && \\
        git clone https://github.com/ornladios/ADIOS2.git {src_path} && \\
        cd {src_path} && \\
        git checkout v{version} && \\
        cmake -B build \\
            -D CMAKE_CXX_STANDARD={cxx_standard} \\
            -D CMAKE_CXX_EXTENSIONS=OFF \\
            -D CMAKE_POSITION_INDEPENDENT_CODE=TRUE \\
            -D BUILD_SHARED_LIBS=ON \\
            -D ADIOS2_USE_Python=OFF \\
            -D ADIOS2_USE_Fortran=OFF \\
            -D ADIOS2_USE_ZeroMQ=OFF \\
            -D BUILD_TESTING=OFF \\
            -D ADIOS2_BUILD_EXAMPLES=OFF \\
            -D ADIOS2_USE_HDF5=OFF \\
            -D ADIOS2_USE_MPI={with_mpi} \\
            -D CMAKE_INSTALL_PREFIX={install_path} {extra_flags} && \\
        cmake --build build -j $(nproc) && \\
        cmake --install build
        """

        modfile = f"""
        #%Module1.0######################################################################
        ##
        ## ADIOS2 @ {mpi_mode} modulefile
        ##
        #################################################################################
        proc ModulesHelp {{ }} {{
            puts stderr \"\\tADIOS2 @ {mpi_mode}\\n\"
        }}

        module-whatis      \"Sets up ADIOS2 @ {mpi_mode}\"    

        conflict           adios2
        {modules}

        set                basedir      {install_path}
        prepend-path       PATH         $basedir/bin
        setenv             ADIOS2_DIR   $basedir

        setenv ADIOS2_USE_MPI {with_mpi}
        """

        return (unindent(script), unindent(modfile))

    else:
        return ("""# skipping Adios2 install""", "")


def InstallNt2pyScript(settings: Settings) -> str:
    if settings.apps.get("nt2py", False):
        prefix = settings.install_prefix
        modules = "\n".join(
            [f"module load {module} && \\" for module in settings.module_loads]
        )
        install_path = f"{prefix}/.venv"

        script = f"""
        # nt2py installation
        {modules}
        rm -rf {install_path} && \\
        python3 -m venv {install_path} && \\
        source {install_path}/bin/activate && \\
        pip install nt2py && \\
        deactivate
        """
        return unindent(script)
    else:
        return """# skipping nt2py install"""


PRESETS = {
    "rusty": {"module_loads": []},
    "stellar": {"module_loads": []},
    "perlmutter": {"module_loads": []},
    "frontier": {"module_loads": []},
    "aurora": {"module_loads": []},
}


def apply_preset(s: Settings, name: str) -> None:
    s.cluster = name
    s.install_prefix = os.path.join(os.path.expanduser("~"), ".entity")
    cluster_preset = PRESETS.get(name, {})
    s.apps["Kokkos"] = True
    s.apps["adios2"] = True
    s.apps["nt2py"] = True
    s.module_loads = cluster_preset.get("module_loads", [])
    s.extra_kokkos_flags = cluster_preset.get("extra_kokkos_flags", [])
    s.extra_adios2_flags = cluster_preset.get("extra_adios2_flags", [])


def on_install_confirmed(settings: Settings) -> None:
    global MESSAGE
    os.makedirs(settings.install_prefix, exist_ok=True)
    kokkos_script, kokkos_modfile = InstallKokkosScriptModfile(settings)
    adios2_script, adios2_modfile = InstallAdios2Script(settings)
    with open(os.path.join(settings.install_prefix, "install.sh"), "w") as f:
        f.write("#!/usr/bin/env bash\n\n")
        f.write(kokkos_script)
        f.write("\n\n")
        f.write(adios2_script)
        f.write("\n")
    if settings.write_modulefiles:
        os.makedirs(os.path.join(settings.install_prefix, "modules"), exist_ok=True)
        if kokkos_modfile != "":
            kokkos_modfile_file = os.path.join(
                settings.install_prefix,
                "modules",
                "kokkos",
                settings.kokkos_backend
                + (
                    f"_{settings.kokkos_arch.strip()}"
                    if settings.kokkos_arch.strip()
                    else ""
                ),
                settings.kokkos_version,
            )
            os.makedirs(os.path.dirname(kokkos_modfile_file), exist_ok=True)
            if os.path.exists(kokkos_modfile_file) and not settings.overwrite:
                raise FileExistsError(
                    f"modulefile {kokkos_modfile_file} already exists and overwrite is disabled"
                )
            with open(kokkos_modfile_file, "w") as f:
                f.write(kokkos_modfile)
        if adios2_modfile != "":
            adios2_modfile_file = os.path.join(
                settings.install_prefix,
                "modules",
                "adios2",
                settings.adios2_mpi,
                settings.adios2_version,
            )
            os.makedirs(os.path.dirname(adios2_modfile_file), exist_ok=True)
            if os.path.exists(adios2_modfile_file) and not settings.overwrite:
                raise FileExistsError(
                    f"modulefile {adios2_modfile_file} already exists and overwrite is disabled"
                )
            with open(adios2_modfile_file, "w") as f:
                f.write(adios2_modfile)

    os.chmod(os.path.join(settings.install_prefix, "install.sh"), 0o755)
    MESSAGE = f"- installation script written to {os.path.join(settings.install_prefix, 'install.sh')}!\n"
    MESSAGE += "  please read and verify it before running.\n\n"
    if settings.write_modulefiles:
        MESSAGE += f"- module files have been written to {os.path.join(settings.install_prefix, 'modules')} directory.\n"
        MESSAGE += f"  add them to your .rc script as `module use --append {os.path.join(settings.install_prefix, 'modules')}`\n\n"

    if settings.apps.get("nt2py", False):
        MESSAGE += (
            "- nt2py installed in a new virtual environment at "
            f"{os.path.join(settings.install_prefix, '.venv')}.\n"
        )
        MESSAGE += "  activate it with `source {}/bin/activate`.\n\n".format(
            os.path.join(settings.install_prefix, ".venv")
        )

    settings_json = os.path.join(settings.install_prefix, "settings.json")
    with open(settings_json, "w") as f:
        f.write(settings.to_json())
    return


@dataclass
class MenuItem:
    label: str
    hint: str = ""
    right: Optional[Callable[[], str]] = None
    on_enter: Optional[Callable[[], None]] = None
    on_space: Optional[Callable[[], None]] = None
    disabled: Optional[Callable[[], bool]] = None


class TuiExitInstall(Exception):
    pass


class App:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.s = Settings()
        if os.path.exists(os.path.join(self.s.install_prefix, "settings.json")):
            with open(os.path.join(self.s.install_prefix, "settings.json"), "r") as f:
                data = json.load(f)
            self.s.from_json(json.dumps(data))

        self.state = "mainmenu"
        self.stack: List[Tuple[str, int]] = []
        self.selected = 0
        self.scroll = 0
        self.message = "use arrows or j/k"

        self.mod_sel = 0
        self.mod_scroll = 0

        self._init_curses()

    def _init_curses(self) -> None:
        curses.curs_set(0)
        self.stdscr.keypad(True)
        curses.noecho()
        curses.cbreak()

        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(PAIR_TITLE, COLOR_TITLE_FG, -1)
            curses.init_pair(PAIR_TEXT, COLOR_TEXT_FG, -1)
            curses.init_pair(PAIR_SELECTED, COLOR_SELECTED_FG, COLOR_SELECTED_BG)
            curses.init_pair(PAIR_HINT, COLOR_HINT_FG, -1)
            curses.init_pair(PAIR_OK, COLOR_OK_FG, -1)
            curses.init_pair(PAIR_ERR, COLOR_ERR_FG, -1)
            curses.init_pair(PAIR_KEY, COLOR_KEY_FG, -1)
            curses.init_pair(PAIR_DIM, COLOR_DIM_FG, -1)

    def cp(self, pair_id: int) -> int:
        return curses.color_pair(pair_id) if curses.has_colors() else 0

    # ----- formatting helpers -----

    def checkbox(self, on: bool) -> str:
        return "[x]" if on else "[ ]"

    def pill(self, on: bool) -> str:
        return "[on]" if on else "[off]"

    def kokkos_right(self) -> str:
        arch = self.s.kokkos_arch.strip() or "-"
        return f"{self.s.kokkos_version} · {self.s.kokkos_backend} · {arch}"

    def adios2_right(self) -> str:
        return f"{self.s.adios2_version} · {self.s.adios2_mpi}"

    # ----- nav stack -----

    def push(self, st: str) -> None:
        self.stack.append((self.state, self.selected))
        self.state = st
        self.selected = 0
        self.scroll = 0
        self.message = ""

    def pop(self) -> None:
        if self.stack:
            self.state, self.selected = self.stack.pop()
        else:
            self.state, self.selected = "mainmenu", 0
        self.scroll = 0
        self.message = ""

    # ----- drawing -----

    def add(self, y: int, x: int, s: str, attr: int = 0) -> None:
        try:
            self.stdscr.addstr(y, x, s, attr)
        except curses.error:
            pass

    def hline(self, y: int) -> None:
        _, w = self.stdscr.getmaxyx()
        try:
            self.stdscr.hline(y, 0, curses.ACS_HLINE, max(0, w - 1))
        except curses.error:
            pass

    def draw_keybar(self, y: int, x: int, pairs: List[Tuple[str, str]]) -> None:
        cur_x = x
        for key, action in pairs:
            self.add(y, cur_x, key, self.cp(PAIR_KEY) | curses.A_BOLD)
            cur_x += len(key)
            self.add(y, cur_x, " ", self.cp(PAIR_DIM))
            cur_x += 1
            self.add(y, cur_x, action, self.cp(PAIR_HINT))
            cur_x += len(action)
            self.add(y, cur_x, "   ", self.cp(PAIR_DIM))
            cur_x += 3

    def breadcrumb(self) -> str:
        if self.state == "mainmenu":
            return "mainmenu"
        if self.state == "custom":
            return "mainmenu › custom install"
        if self.state == "dependencies":
            return "mainmenu › custom install › dependencies"
        if self.state == "versions":
            return "mainmenu › custom install › versions"
        if self.state == "options":
            return "mainmenu › custom install › options"
        if self.state == "cluster":
            return "mainmenu › cluster-specific"
        if self.state == "preset_applied":
            return f"mainmenu › cluster-specific › {self.s.cluster}"
        return "mainmenu"

    def draw_menu(self, title: str, prompt: str, items: List[MenuItem]) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()

        self.add(0, 2, title, self.cp(PAIR_TITLE) | curses.A_BOLD)
        bc = self.breadcrumb()
        self.add(0, max(2, w - 2 - len(bc)), bc, self.cp(PAIR_DIM))

        self.draw_keybar(
            1,
            2,
            [
                ("↑/↓/j/k", "move"),
                ("enter", "select"),
                ("space", "toggle/cycle"),
                ("b", "back"),
                ("q", "quit"),
            ],
        )
        self.hline(2)

        status1 = f"cluster: {self.s.cluster}    write modulefiles: {self.pill(self.s.write_modulefiles)}    module loads: {len(self.s.module_loads)}"
        status2 = (
            f"prefix: {self.s.install_prefix}    dependencies: {self.s.apps_summary()}"
        )
        self.add(3, 2, status1[: w - 4], self.cp(PAIR_TEXT))
        self.add(4, 2, status2[: w - 4], self.cp(PAIR_TEXT))
        self.hline(5)

        self.add(6, 2, prompt[: w - 4], self.cp(PAIR_TEXT) | curses.A_BOLD)

        list_y = 8
        footer_h = 3
        view_h = max(1, h - list_y - footer_h)
        n = len(items)

        if n == 0:
            self.add(list_y, 2, "(empty)", self.cp(PAIR_HINT))
        else:
            self.selected = max(0, min(self.selected, n - 1))

            if self.selected < self.scroll:
                self.scroll = self.selected
            if self.selected >= self.scroll + view_h:
                self.scroll = self.selected - view_h + 1
            self.scroll = max(0, min(self.scroll, max(0, n - view_h)))

            shown = items[self.scroll : self.scroll + view_h]

            for i, it in enumerate(shown):
                idx = self.scroll + i
                sel = idx == self.selected
                dis = bool(it.disabled and it.disabled())

                row_attr = (
                    self.cp(PAIR_SELECTED) | curses.A_BOLD
                    if sel
                    else (self.cp(PAIR_DIM) if dis else self.cp(PAIR_TEXT))
                )
                self.add(list_y + i, 2, f"  {it.label}"[: w - 4], row_attr)

                if it.right:
                    rt = (it.right() or "").strip()
                    if rt:
                        rt = rt[: max(0, w - 6)]
                        x = max(2, w - 2 - len(rt))
                        rt_attr = (
                            row_attr
                            if sel
                            else (self.cp(PAIR_HINT) if not dis else self.cp(PAIR_DIM))
                        )
                        self.add(list_y + i, x, rt, rt_attr)

                if sel and it.hint:
                    self.add(
                        list_y + i,
                        min(w - 4, 30),
                        f"    {it.hint}"[: w - 4],
                        self.cp(PAIR_HINT),
                    )

        self.hline(h - 3)
        msg = self.message or ""
        if msg:
            is_err = msg.startswith("error")
            attr = (self.cp(PAIR_ERR) if is_err else self.cp(PAIR_OK)) | curses.A_BOLD
            self.add(h - 2, 2, msg[: w - 4], attr)
        self.stdscr.refresh()

    # ----- modals -----

    def input_box(self, title: str, prompt: str, initial: str) -> Optional[str]:
        h, w = self.stdscr.getmaxyx()
        win_h, win_w = 9, min(86, max(46, w - 6))
        top, left = max(0, (h - win_h) // 2), max(0, (w - win_w) // 2)

        win = curses.newwin(win_h, win_w, top, left)
        win.keypad(True)
        win.border()

        win.addstr(1, 2, title[: win_w - 4], self.cp(PAIR_TITLE) | curses.A_BOLD)
        win.addstr(2, 2, prompt[: win_w - 4], self.cp(PAIR_TEXT))

        buf = list(initial)
        curses.curs_set(1)

        while True:
            win.addstr(4, 2, " " * (win_w - 4), self.cp(PAIR_TEXT))
            text = "".join(buf)
            if len(text) > win_w - 4:
                text = text[-(win_w - 4) :]
            win.addstr(4, 2, text, self.cp(PAIR_TEXT) | curses.A_BOLD)
            win.addstr(6, 2, "enter=ok   esc=cancel", self.cp(PAIR_DIM))
            win.refresh()

            ch = win.getch()
            if ch == 27:
                curses.curs_set(0)
                return None
            if ch in (curses.KEY_ENTER, 10, 13):
                curses.curs_set(0)
                return "".join(buf).strip()
            if ch in (curses.KEY_BACKSPACE, 127, 8):
                if buf:
                    buf.pop()
            elif 32 <= ch <= 126:
                buf.append(chr(ch))

    def confirm_install(self) -> bool:
        arch = self.s.kokkos_arch.strip() or "-"
        lines = [
            f"cluster: {self.s.cluster}",
            f"overwrite existing files: {self.pill(self.s.overwrite)}",
            f"write modulefiles: {self.pill(self.s.write_modulefiles)}",
            f"module loads: {len(self.s.module_loads)}",
            f"prefix: {self.s.install_prefix}",
            f"dependencies: {self.s.apps_summary()}",
            f"kokkos: {self.s.kokkos_version} · {self.s.kokkos_backend} · {arch}",
            f"adios2: {self.s.adios2_version} · {self.s.adios2_mpi}",
            "",
            "confirm install?",
        ]

        h, w = self.stdscr.getmaxyx()
        win_h, win_w = min(16, max(10, h - 6)), min(94, max(52, w - 6))
        top, left = max(0, (h - win_h) // 2), max(0, (w - win_w) // 2)

        win = curses.newwin(win_h, win_w, top, left)
        win.keypad(True)
        win.border()
        win.addstr(1, 2, "confirm", self.cp(PAIR_TITLE) | curses.A_BOLD)

        y = 3
        for ln in lines[: win_h - 6]:
            win.addstr(y, 2, ln[: win_w - 4], self.cp(PAIR_TEXT))
            y += 1

        win.addstr(win_h - 3, 2, "y=yes   n=no", self.cp(PAIR_DIM))
        win.refresh()

        while True:
            ch = win.getch()
            if ch in (ord("y"), ord("Y")):
                return True
            if ch in (ord("n"), ord("N"), 27):
                return False

    # ----- helpers -----

    def cycle(self, current: str, options: List[str]) -> str:
        if current not in options:
            return options[0]
        i = options.index(current)
        return options[(i + 1) % len(options)]

    # ----- module editor -----

    def module_editor(self) -> None:
        while True:
            self.stdscr.erase()
            h, w = self.stdscr.getmaxyx()

            self.add(0, 2, "module lines", self.cp(PAIR_TITLE) | curses.A_BOLD)
            self.draw_keybar(
                1,
                2,
                [
                    ("↑/↓/j/k", "move"),
                    ("enter", "edit"),
                    ("a", "add"),
                    ("d", "delete"),
                    ("u/m", "reorder"),
                    ("b", "back"),
                ],
            )
            self.hline(2)

            self.add(3, 2, f"lines: {len(self.s.module_loads)}", self.cp(PAIR_TEXT))
            self.hline(4)

            lines = self.s.module_loads
            n = len(lines)
            list_y = 6
            view_h = max(1, h - list_y - 3)

            if n == 0:
                self.add(list_y, 2, "(empty) press a to add", self.cp(PAIR_HINT))
            else:
                self.mod_sel = max(0, min(self.mod_sel, n - 1))
                if self.mod_sel < self.mod_scroll:
                    self.mod_scroll = self.mod_sel
                if self.mod_sel >= self.mod_scroll + view_h:
                    self.mod_scroll = self.mod_sel - view_h + 1
                self.mod_scroll = max(0, min(self.mod_scroll, max(0, n - view_h)))

                shown = lines[self.mod_scroll : self.mod_scroll + view_h]
                for i, ln in enumerate(shown):
                    idx = self.mod_scroll + i
                    sel = idx == self.mod_sel
                    attr = (
                        self.cp(PAIR_SELECTED) | curses.A_BOLD
                        if sel
                        else self.cp(PAIR_TEXT)
                    )
                    self.add(list_y + i, 2, f"  {ln}"[: w - 4], attr)

            self.hline(h - 3)
            self.add(
                h - 2,
                2,
                "tip: example: cuda/12.9"[: w - 4],
                self.cp(PAIR_HINT),
            )
            self.stdscr.refresh()

            ch = self.stdscr.getch()
            if ch in (ord("q"), ord("Q"), ord("b"), 8, 127):
                return

            n = len(self.s.module_loads)
            self.mod_sel = 0 if n == 0 else max(0, min(self.mod_sel, n - 1))

            if ch in (curses.KEY_UP, ord("k"), ord("K")) and n:
                self.mod_sel = (self.mod_sel - 1) % n
                continue
            if ch in (curses.KEY_DOWN, ord("j"), ord("J")) and n:
                self.mod_sel = (self.mod_sel + 1) % n
                continue

            if ch in (ord("a"), ord("A")):
                val = self.input_box("add module line", "example: cuda/12.9", "")
                if val:
                    self.s.module_loads.append(val)
                    self.mod_sel = len(self.s.module_loads) - 1
                continue

            if ch in (ord("d"), ord("D")):
                if n == 0:
                    continue
                val = self.input_box("delete line", "type 'delete' to confirm:", "")
                if val == "delete":
                    del self.s.module_loads[self.mod_sel]
                    self.mod_sel = max(
                        0, min(self.mod_sel, len(self.s.module_loads) - 1)
                    )
                continue

            if ch in (ord("u"), ord("U")):
                if n >= 2 and self.mod_sel > 0:
                    i = self.mod_sel
                    self.s.module_loads[i - 1], self.s.module_loads[i] = (
                        self.s.module_loads[i],
                        self.s.module_loads[i - 1],
                    )
                    self.mod_sel -= 1
                continue

            if ch in (ord("m"), ord("M")):
                if n >= 2 and self.mod_sel < n - 1:
                    i = self.mod_sel
                    self.s.module_loads[i + 1], self.s.module_loads[i] = (
                        self.s.module_loads[i],
                        self.s.module_loads[i + 1],
                    )
                    self.mod_sel += 1
                continue

            if ch in (curses.KEY_ENTER, 10, 13):
                if n == 0:
                    continue
                cur = self.s.module_loads[self.mod_sel]
                val = self.input_box("edit module line", "edit the selected line:", cur)
                if val is not None and val.strip():
                    self.s.module_loads[self.mod_sel] = val.strip()
                continue

    # ----- menus -----

    def versions_menu(self) -> Tuple[str, str, List[MenuItem]]:
        def edit_kokkos():
            val = self.input_box(
                "kokkos version", "enter version/tag:", self.s.kokkos_version
            )
            if val:
                self.s.kokkos_version = val.strip()

        def edit_adios2():
            val = self.input_box(
                "adios2 version", "enter version/tag:", self.s.adios2_version
            )
            if val:
                self.s.adios2_version = val.strip()

        return (
            "versions",
            "set versions:",
            [
                MenuItem(
                    "kokkos version",
                    "enter to edit",
                    right=lambda: self.s.kokkos_version,
                    on_enter=edit_kokkos,
                ),
                MenuItem(
                    "adios2 version",
                    "enter to edit",
                    right=lambda: self.s.adios2_version,
                    on_enter=edit_adios2,
                ),
                MenuItem("back", "return", on_enter=self.pop),
            ],
        )

    def options_menu(self) -> Tuple[str, str, List[MenuItem]]:
        def cycle_kokkos():
            self.s.kokkos_backend = self.cycle(self.s.kokkos_backend, KOKKOS_BACKENDS)

        def edit_kokkos_arch():
            val = self.input_box(
                "kokkos arch", "enter arch text (free-form):", self.s.kokkos_arch
            )
            if val is not None:
                self.s.kokkos_arch = val.strip()

        def cycle_adios2():
            self.s.adios2_mpi = self.cycle(self.s.adios2_mpi, ADIOS2_MPI_MODES)

        return (
            "options",
            "set build options:",
            [
                MenuItem(
                    "kokkos backend",
                    "space cycles: cpu/cuda/hip/sycl",
                    right=lambda: self.s.kokkos_backend,
                    on_enter=cycle_kokkos,
                    on_space=cycle_kokkos,
                    disabled=lambda: not self.s.apps.get("Kokkos", False),
                ),
                MenuItem(
                    "kokkos arch",
                    "enter to edit (optional)",
                    right=lambda: (self.s.kokkos_arch.strip() or "-"),
                    on_enter=edit_kokkos_arch,
                    disabled=lambda: not self.s.apps.get("Kokkos", False),
                ),
                MenuItem(
                    "adios2 mpi",
                    "space cycles: non-mpi/mpi",
                    right=lambda: self.s.adios2_mpi,
                    on_enter=cycle_adios2,
                    on_space=cycle_adios2,
                    disabled=lambda: not self.s.apps.get("adios2", False),
                ),
                MenuItem("back", "return", on_enter=self.pop),
            ],
        )

    def menu_main(self) -> Tuple[str, str, List[MenuItem]]:
        return (
            "entity deps",
            "main menu:",
            [
                MenuItem(
                    "custom install",
                    "edit settings then install",
                    on_enter=lambda: self.push("custom"),
                ),
                MenuItem(
                    "cluster-specific",
                    "apply a cluster-specific preset (editable)",
                    on_enter=lambda: self.push("cluster"),
                ),
                MenuItem("exit", "", on_enter=lambda: setattr(self, "state", "exit")),
            ],
        )

    def menu_custom(self) -> Tuple[str, str, List[MenuItem]]:
        def toggle_write_modulefiles():
            self.s.write_modulefiles = not self.s.write_modulefiles

        def toggle_overwrite():
            self.s.overwrite = not self.s.overwrite

        def edit_prefix():
            val = self.input_box(
                "install location", "enter install prefix:", self.s.install_prefix
            )
            if val:
                self.s.install_prefix = os.path.expanduser(val.strip())

        def go_apps():
            self.push("dependencies")

        def go_versions():
            self.push("versions")

        def go_options():
            self.push("options")

        def do_install():
            if not self.confirm_install():
                self.message = "cancelled."
                return
            on_install_confirmed(self.s)
            raise TuiExitInstall

        return (
            "custom install",
            "settings:",
            [
                MenuItem(
                    "overwrite existing files",
                    "whether to overwrite existing files",
                    right=lambda: "enabled" if self.s.overwrite else "disabled",
                    on_enter=toggle_overwrite,
                    on_space=toggle_overwrite,
                ),
                MenuItem(
                    "write modulefiles",
                    "whether to create module files",
                    right=lambda: "enabled" if self.s.write_modulefiles else "disabled",
                    on_enter=toggle_write_modulefiles,
                    on_space=toggle_write_modulefiles,
                ),
                MenuItem(
                    "module load lines",
                    "add/remove modules to load",
                    right=lambda: f"{len(self.s.module_loads)} entry(s)",
                    on_enter=self.module_editor,
                ),
                MenuItem(
                    "install location",
                    "root location where modules and dependencies are installed",
                    right=lambda: self.s.install_prefix,
                    on_enter=edit_prefix,
                ),
                MenuItem(
                    "dependencies to install",
                    "select which dependencies to install",
                    right=lambda: self.s.apps_summary(),
                    on_enter=go_apps,
                ),
                MenuItem(
                    "versions",
                    "edit dependency versions",
                    right=lambda: " · ".join(
                        [
                            a
                            for (a, ae) in zip(
                                [
                                    f"kokkos {self.s.kokkos_version}",
                                    f"adios2 {self.s.adios2_version}",
                                ],
                                [
                                    self.s.apps.get(app, False)
                                    for app in ["Kokkos", "adios2"]
                                ],
                            )
                            if ae
                        ]
                    ),
                    on_enter=go_versions,
                ),
                MenuItem(
                    "options",
                    "pick backends/architectures/mpi",
                    right=lambda: " · ".join(
                        [
                            a
                            for (a, ae) in zip(
                                [
                                    f"kokkos {self.s.kokkos_backend}/{self.s.kokkos_arch.strip() or '-'}",
                                    f"adios2 {self.s.adios2_mpi}",
                                ],
                                [
                                    self.s.apps.get(app, False)
                                    for app in ["Kokkos", "adios2"]
                                ],
                            )
                            if ae
                        ]
                    ),
                    on_enter=go_options,
                ),
                MenuItem("install", "", on_enter=do_install),
                MenuItem("back", "", on_enter=self.pop),
            ],
        )

    def menu_apps(self) -> Tuple[str, str, List[MenuItem]]:
        def toggle(k: str):
            self.s.apps[k] = not self.s.apps.get(k, False)

        return (
            "dependencies",
            "select the dependencies:",
            [
                MenuItem(
                    f"{self.checkbox(self.s.apps.get('Kokkos', False))} kokkos",
                    "",
                    on_enter=lambda: toggle("Kokkos"),
                    on_space=lambda: toggle("Kokkos"),
                    right=self.kokkos_right,
                ),
                MenuItem(
                    f"{self.checkbox(self.s.apps.get('adios2', False))} adios2",
                    "",
                    on_enter=lambda: toggle("adios2"),
                    on_space=lambda: toggle("adios2"),
                    right=self.adios2_right,
                ),
                MenuItem(
                    f"{self.checkbox(self.s.apps.get('nt2py', False))} nt2py",
                    "",
                    on_enter=lambda: toggle("nt2py"),
                    on_space=lambda: toggle("nt2py"),
                ),
                MenuItem("back", "", on_enter=self.pop),
            ],
        )

    def menu_cluster(self) -> Tuple[str, str, List[MenuItem]]:
        def choose(name: str):
            apply_preset(self.s, name)
            self.push("custom")

        return (
            "cluster-specific",
            "pick a preset:",
            [
                MenuItem("rusty", "apply preset", on_enter=lambda: choose("rusty")),
                MenuItem("stellar", "apply preset", on_enter=lambda: choose("stellar")),
                MenuItem(
                    "perlmutter", "apply preset", on_enter=lambda: choose("perlmutter")
                ),
                MenuItem(
                    "frontier", "apply preset", on_enter=lambda: choose("frontier")
                ),
                MenuItem("aurora", "apply preset", on_enter=lambda: choose("aurora")),
                MenuItem("back", "", on_enter=self.pop),
            ],
        )

    def get_menu(self) -> Tuple[str, str, List[MenuItem]]:
        if self.state == "mainmenu":
            return self.menu_main()
        if self.state == "custom":
            return self.menu_custom()
        if self.state == "dependencies":
            return self.menu_apps()
        if self.state == "versions":
            return self.versions_menu()
        if self.state == "options":
            return self.options_menu()
        if self.state == "cluster":
            return self.menu_cluster()
        self.state = "mainmenu"
        return self.menu_main()

    # ----- navigation -----

    def is_disabled(self, it: MenuItem) -> bool:
        return bool(it.disabled and it.disabled())

    def move_sel(self, items: List[MenuItem], delta: int) -> None:
        if not items:
            return
        n = len(items)
        start = self.selected
        for _ in range(n):
            self.selected = (self.selected + delta) % n
            if not self.is_disabled(items[self.selected]):
                return
        self.selected = start

    def activate(self, items: List[MenuItem], enter: bool) -> None:
        if not items:
            return
        it = items[self.selected]
        if self.is_disabled(it):
            self.message = "error: option disabled."
            return
        fn = it.on_enter if enter else it.on_space
        if fn:
            fn()

    # ----- loop -----

    def run(self) -> None:
        while True:
            if self.state == "exit":
                return

            title, prompt, items = self.get_menu()
            self.draw_menu(title, prompt, items)

            ch = self.stdscr.getch()

            if ch in (ord("q"), ord("Q")):
                self.state = "exit"
                continue

            if ch in (ord("b"), 8, 127):
                self.pop()
                continue

            if ch in (curses.KEY_UP, ord("k"), ord("K")):
                self.move_sel(items, -1)
                continue

            if ch in (curses.KEY_DOWN, ord("j"), ord("J")):
                self.move_sel(items, +1)
                continue

            if ch in (curses.KEY_ENTER, 10, 13):
                self.activate(items, enter=True)
                continue

            if ch == ord(" "):
                self.activate(items, enter=False)
                continue


def _wrapper_capture(stdscr) -> None:
    app = App(stdscr)
    try:
        app.run()
    except TuiExitInstall:
        on_install_confirmed(app.s)
        raise


if __name__ == "__main__":
    try:
        curses.wrapper(_wrapper_capture)
        raise SystemExit(0)
    except TuiExitInstall:
        print(MESSAGE)
        raise SystemExit(0)
    except KeyboardInterrupt:
        raise SystemExit(130)
