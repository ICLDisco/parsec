# PaRSEC syntax for Vim / Neovim

Syntax highlighting for PaRSEC JDF files (`*.jdf`, the PTG input language),
plus highlighting of the DTD (Dynamic Task Discovery) API in C and C++ files.

| File | What it does |
|---|---|
| `ftdetect/jdf.vim` | Sets the filetype of `*.jdf` files to `parsec`. |
| `syntax/parsec.vim` | JDF syntax. C is highlighted only inside `extern "C" %{ %}`, inline `%{ %}` and `BODY ... END`. |
| `ftplugin/parsec.vim` | `commentstring` and `comments` for `//` and `/* */`. |
| `after/syntax/c.vim` | Adds the DTD names (`parsec_dtd_*`, `PARSEC_INOUT`, ...) on top of the C syntax. Optional. |

## Install

Use one of the options below. In both cases, replace `<parsec>` with the path
of your PaRSEC source tree.

### Option 1: as a package (Vim 8+ and Neovim)

Link the directory into a `start` package. The `after/` directory of a package
is added to the runtime path automatically, so the DTD highlighting is included.

Vim:

```sh
mkdir -p ~/.vim/pack/parsec/start
ln -s <parsec>/tools/vim_syntax ~/.vim/pack/parsec/start/parsec-jdf
```

Neovim:

```sh
mkdir -p ~/.local/share/nvim/site/pack/parsec/start
ln -s <parsec>/tools/vim_syntax ~/.local/share/nvim/site/pack/parsec/start/parsec-jdf
```

### Option 2: copy the files

Copy each directory into your Vim runtime directory (`~/.vim`, or
`~/.config/nvim` for Neovim). The `after/` directory must stay named `after`.

```sh
cp -r <parsec>/tools/vim_syntax/{ftdetect,syntax,ftplugin,after} ~/.vim/
```

If `~/.vim` already has an `after/syntax/c.vim`, merge the two files instead of
overwriting it.

### Only the JDF part

The DTD file is independent. To skip it, do not install `after/`.

## Check

Open a `.jdf` file (for example `examples/Ex05_Broadcast.jdf`) and run:

```vim
:set filetype?
```

It must answer `filetype=parsec`. If it does not, make sure filetype detection
and syntax highlighting are on (`filetype plugin on` and `syntax on` in your
`vimrc`).

To check the DTD highlighting, open a file from `tests/dsl/dtd/`:
`parsec_dtd_insert_task` and `PARSEC_INOUT` use different colors than plain
function names and macros.
