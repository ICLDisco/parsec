# PaRSEC syntax for Cursor / VS Code

Syntax highlighting for PaRSEC JDF files (`*.jdf`, the PTG input language),
plus highlighting of the DTD (Dynamic Task Discovery) API in C, C++ and CUDA
files. This is a declarative extension: it only contains grammars, no code.

| File | What it does |
|---|---|
| `package.json` | Registers the `jdf` language for `*.jdf` and the two grammars below. |
| `syntaxes/jdf.tmLanguage.json` | JDF grammar. C is highlighted only inside `extern "C" %{ %}`, inline `%{ %}` and `BODY ... END`. |
| `syntaxes/parsec-dtd.injection.tmLanguage.json` | Adds the DTD names (`parsec_dtd_*`, `PARSEC_INOUT`, ...) on top of C, C++, CUDA and the C inside JDF bodies. |
| `language-configuration.json` | Comment toggling (`//`, `/* */`), brackets and auto-closing pairs. |

## Install

In all cases, replace `<parsec>` with the path of your PaRSEC source tree.
Reload the window afterwards (Command Palette: "Developer: Reload Window").

### Option 1: install from the folder

1. Open the Command Palette.
2. Run "Developer: Install Extension from Location...".
3. Select `<parsec>/tools/vscode_syntax`.

### Option 2: link the folder into the extensions directory

The folder name must be `<publisher>.<name>-<version>`:

```sh
ln -s <parsec>/tools/vscode_syntax ~/.cursor/extensions/icldisco.parsec-jdf-0.1.0
```

When working on a remote machine over SSH, the extension has to be on the
remote side, so use the remote server directory instead:

```sh
ln -s <parsec>/tools/vscode_syntax ~/.cursor-server/extensions/icldisco.parsec-jdf-0.1.0
```

VS Code uses `~/.vscode/extensions` and `~/.vscode-server/extensions`
instead of the Cursor directories.

To uninstall, remove the link (or use the Extensions view) and reload.

## Check

- Open a `.jdf` file (for example `examples/Ex05_Broadcast.jdf`). The language
  shown in the status bar must be "PaRSEC JDF".
- Run "Developer: Inspect Editor Tokens and Scopes" on a token to see its
  scopes. For example, a task class name has the scope
  `entity.name.function.jdf`, and the `parsec_dtd_insert_task` call in a file
  from `tests/dsl/dtd/` has `support.function.parsec-dtd.c`.

## Notes

- Colors come from your theme. The grammars use the standard TextMate scope
  names, so any theme colors them.
- To disable the DTD highlighting in C and C++ while keeping JDF, delete the
  `parsec-dtd.injection` entry from `contributes.grammars` in `package.json`.
- In C++, a DTD function call that has no whitespace before it (for example
  `(parsec_dtd_foo(`) may keep the C++ color. C is not affected.
