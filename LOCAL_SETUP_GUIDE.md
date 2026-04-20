# M4 Lab: Local Setup Guide

If the wifi or Codespaces are not cooperating, you can run this lab entirely on your own computer. This guide walks you through installing Julia and all required packages.

**Estimated setup time:** 15–20 minutes (mostly downloads)

---

## macOS

### Step 1: Install Julia

**Option A — Homebrew (recommended if you have Homebrew):**
```bash
brew install julia
```

**Option B — Download directly:**
1. Go to [https://julialang.org/downloads/](https://julialang.org/downloads/)
2. Download the **macOS** installer (Apple Silicon if you have M1/M2/M3/M4 chip; Intel x86 otherwise)
3. Open the `.dmg` file, drag Julia to Applications
4. Open Terminal and verify:
   ```bash
   julia --version
   ```
   You should see `julia version 1.11.x` or similar.

### Step 2: Download the Lab Files

Download or clone the lab repository:
```bash
git clone https://github.com/utsw-hdsb/pubh5106-m4-lab.git
cd pubh5106-m4-lab
```

Or if git is not available, download the ZIP from the GitHub repository page and unzip it.

### Step 3: Install Julia Packages and Jupyter

Open Terminal, navigate to the lab folder, and start Julia:
```bash
cd pubh5106-m4-lab
julia --project=.
```

At the Julia prompt (`julia>`), run:
```julia
using Pkg
Pkg.instantiate()    # installs all packages from Project.toml
using IJulia
notebook()           # launches Jupyter in your browser
```

The first time you run `using IJulia`, Julia will ask if you want to install Jupyter via Conda. **Type `y` and press Enter.** This installs a private copy of Python + Jupyter managed by Julia — you do NOT need to install Python separately.

The `Pkg.instantiate()` step downloads and compiles packages. This takes 5–10 minutes the first time. Subsequent launches are fast.

### Step 4: Open the Notebook

Once Jupyter launches in your browser, click on `M4_lab.ipynb` to open the lab notebook. Make sure the kernel says **Julia 1.x** (not Python).

---

## Windows

### Step 1: Install Julia

1. Go to [https://julialang.org/downloads/](https://julialang.org/downloads/)
2. Download the **Windows** installer (64-bit, `.exe`)
3. Run the installer. **Important:** Check the box **"Add Julia to PATH"** during installation
4. Open a **new** Command Prompt or PowerShell window and verify:
   ```
   julia --version
   ```
   You should see `julia version 1.11.x` or similar. If you get "command not found," close and reopen the terminal — the PATH change requires a new window.

### Step 2: Download the Lab Files

**Option A — Git (if installed):**
```
git clone https://github.com/utsw-hdsb/pubh5106-m4-lab.git
cd pubh5106-m4-lab
```

**Option B — Download ZIP:**
1. Go to the GitHub repository page for the lab
2. Click the green **Code** button → **Download ZIP**
3. Unzip the folder to a location you can find (e.g., `C:\Users\YourName\Documents\pubh5106-m4-lab`)
4. Open Command Prompt and navigate there:
   ```
   cd C:\Users\YourName\Documents\pubh5106-m4-lab
   ```

### Step 3: Install Julia Packages and Jupyter

Start Julia from the lab folder:
```
julia --project=.
```

At the Julia prompt (`julia>`), run:
```julia
using Pkg
Pkg.instantiate()    # installs all packages from Project.toml
using IJulia
notebook()           # launches Jupyter in your browser
```

When Julia asks if you want to install Jupyter via Conda, **type `y` and press Enter.** This installs a private copy of Python + Jupyter — no separate Python installation needed.

The `Pkg.instantiate()` step takes 5–10 minutes the first time (downloading and compiling packages). Be patient — Julia compiles packages ahead of time so they run fast later.

### Step 4: Open the Notebook

Once Jupyter launches in your browser, click on `M4_lab.ipynb` to open the lab notebook. Make sure the kernel says **Julia 1.x** (not Python).

---

## Troubleshooting

### "julia: command not found" (Mac) or "'julia' is not recognized" (Windows)

Julia is not on your PATH. 
- **Mac:** Add to your shell profile: `export PATH="/Applications/Julia-1.11.app/Contents/Resources/julia/bin:$PATH"` (adjust version number as needed), then restart Terminal.
- **Windows:** Re-run the installer and check "Add to PATH," or manually add `C:\Users\YourName\AppData\Local\Programs\Julia-1.11.x\bin` to your system PATH.

### "Pkg.instantiate() is taking forever"

This is normal the first time. Julia pre-compiles packages so they load fast in future sessions. If it exceeds 15 minutes, check your internet connection — packages are downloaded from the Julia package registry.

### IJulia asks about installing Jupyter

Say **yes** (`y`). Julia will download and install a private Miniconda with Python and Jupyter. This does not interfere with any existing Python installation on your system.

### Kernel dies or notebook won't start

Make sure you launched Julia with `--project=.` from the lab directory. This activates the correct package environment. If you launched Julia from a different directory, the packages won't be found.

### Plotting errors (CairoMakie / GraphMakie)

These packages require compilation on first use, which can take a few minutes. If you see precompilation messages, wait for them to finish. On some Windows systems, you may need to install the [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

### "No matching method" or package errors

Run these commands in Julia to ensure everything is up to date:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()
```

---

## Do I Need Python?

**No.** Julia manages its own Python installation via the Conda.jl package. When you install IJulia, it automatically sets up a private Python + Jupyter environment. This is completely separate from any Python you may have installed on your system (Anaconda, Miniconda, system Python, etc.).

If you already have Jupyter installed and want Julia to use it instead of installing its own copy, you can set:
```julia
ENV["JUPYTER"] = "/path/to/your/jupyter"
using Pkg
Pkg.build("IJulia")
```

But for most students, the default (let Julia install its own Jupyter) is the simplest path.

---

## Quick Reference

| Step | Mac | Windows |
|------|-----|---------|
| Install Julia | `brew install julia` or download from julialang.org | Download .exe from julialang.org, check "Add to PATH" |
| Get lab files | `git clone ...` or download ZIP | `git clone ...` or download ZIP |
| Install packages | `julia --project=.` then `Pkg.instantiate()` | Same |
| Start notebook | `using IJulia; notebook()` | Same |
| Python needed? | No — Julia installs its own | No — Julia installs its own |
