# CRITICAL RULES

- Scan the existing codebase and reuse existing functions wherever possible.
- Keep all imports within functions unless they must be mocked in a test.
- If an import is small, performative, and significantly reduces needs for new code, use the library.
- Code files must be under 200 lines of code.
- Write short Sphinx docstrings as a single description line and a line for each parameter.
- On first line of docstrings use \n instead of line break.
- Variable names must be `snake_case` sequence of descriptive words <=5 letters long.
- Keep labels consistent across the entire project.
- In commit messages: use `+` for code adds, `-` for code subtractions, `~` for refactors/fixes.
- Write full variable names at all times.
- Never exceed 200 lines of code in a single file.
- Use descriptive variable names instead of comments.
- No abbreviations.
- No empty docstring lines.
- No inline comments.
- No emoji.
- No global variables.
- No semantic commit messages.
