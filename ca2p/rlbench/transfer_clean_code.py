# transfer_clean_code.py
import ast
import io
import tokenize
import textwrap
import re


def _strip_comments_preserve_layout(source: str) -> str:
    """Remove only comments via tokenization while preserving line breaks, whitespace, and strings."""
    sio = io.StringIO(source)
    out = []
    for tok in tokenize.generate_tokens(sio.readline):
        if tok.type == tokenize.COMMENT:
            continue
        out.append((tok.type, tok.string))
    return tokenize.untokenize(out)


def _drop_inline_string_after_run_task_def(src: str) -> str:
    """
    'def run_task(...):' immediately followed by a single-quoted
    remove the odd inline string block only in that case.
    (leave triple quotes and regular docstrings untouched)
    """
    lines = src.splitlines(keepends=True)
    out = []
    i = 0

    def is_run_task_def(line: str) -> bool:
        return re.match(r'^\s*def\s+run_task\s*\(.*\):\s*(#.*)?$', line) is not None

    while i < len(lines):
        line = lines[i]
        out.append(line)

        if is_run_task_def(line) and (i + 1) < len(lines):
            nxt = lines[i + 1]
            s = nxt.lstrip()


            if s.startswith('"""') or s.startswith("'''"):
                i += 1
                continue


            if s and s[0] in ('"', "'") and not s.startswith(s[0] * 3):
                q = s[0]


                if s.startswith(q + r"\n") and s.rstrip().endswith(q):

                    i += 2
                    continue


                j = i + 1
                removed_block = False
                if s.strip() == q:
                    j += 1
                    while j < len(lines):
                        t = lines[j]
                        if t.strip() == q:
                            j += 1
                            removed_block = True
                            break
                        j += 1
                if removed_block:
                    i = j
                    continue

        i += 1

    return "".join(out)


class HeaderHelperRemover(ast.NodeTransformer):
    top_level_funcs = {
        "_quat_mul", "_quat_from_euler", "_yaw_from_quat",
        "_get_object_constraints", "_axis_from_vector"
    }
    top_level_vars = {
        "_flip_dict", "_approach_axis_dict",
    }

    def visit_Assign(self, node):
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if names & self.top_level_vars:
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name in self.top_level_funcs:
            return None

        if node.name == "run_task":
            new_body = []
            for stmt in node.body:

                if isinstance(stmt, ast.Assign):
                    if any(isinstance(t, ast.Name) and t.id == "common_args" for t in stmt.targets):
                        continue

                if isinstance(stmt, ast.FunctionDef) and stmt.name == "run_action":
                    continue
                new_body.append(stmt)
            node.body = new_body
        return self.generic_visit(node)


class ImportRemover(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None


class DocstringRemover(ast.NodeTransformer):
    """Remove only leading docstrings of modules, classes, or functions."""
    @staticmethod
    def _is_str_const(expr):
        return (
            isinstance(expr, ast.Expr)
            and isinstance(getattr(expr, "value", None), ast.Constant)
            and isinstance(expr.value.value, str)
        )

    def _strip_leading_doc(self, node):
        if getattr(node, "body", None) and self._is_str_const(node.body[0]):
            node.body.pop(0)
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip_leading_doc(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip_leading_doc(node)

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        return self._strip_leading_doc(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip_leading_doc(node)


def _parse_as_module(src: str):
    return ast.parse(src, mode="exec")


def _parse_as_snippet_best_effort(src: str):
    """
    Wrap the code snippet in a function and parse it.
    If parsing fails, drop lines from the end and retry.
    On success return (tree, True, number_of_lines_used).
    On failure return (None, True, 0).
    """
    lines = src.splitlines()
    while lines:
        body = "\n".join(lines)
        wrapped = "def __snippet__():\n" + textwrap.indent(body, "    ")
        try:
            tree = ast.parse(wrapped, mode="exec")
            return tree, True, len(lines)
        except SyntaxError:

            lines = lines[:-1]
    return None, True, 0


def remove_comments_and_docstrings(source: str) -> str:
    """
    Safe cleaner:
    1) remove only comments (preserve strings/indentation)
    2) optionally remove the odd single-quoted block right after run_task
    3) parse the module → if it fails, wrap the snippet and retry after trimming from the end
    4) remove docstrings and apply custom transforms
    5) if snippet mode was used restore only the __snippet__ body
    """
    if not isinstance(source, str):
        source = str(source)


    src = source.replace("\r\n", "\n").replace("\r", "\n")

    src = src.replace("warnings.filterwarnings('error', category=RuntimeWarning)", "")


    no_comments = _strip_comments_preserve_layout(src)


    cleaned = _drop_inline_string_after_run_task_def(no_comments)


    snippet_mode = False
    try:
        tree = _parse_as_module(cleaned)
    except (IndentationError, SyntaxError):
        tree, snippet_mode, used = _parse_as_snippet_best_effort(cleaned)
        if tree is None:

            return no_comments if no_comments.strip() else src


    tree = HeaderHelperRemover().visit(tree)
    tree = DocstringRemover().visit(tree)
    tree = ImportRemover().visit(tree)
    ast.fix_missing_locations(tree)


    if not snippet_mode:
        try:
            return ast.unparse(tree)
        except Exception:

            return cleaned


    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "__snippet__"), None)
    if not func or not getattr(func, "body", None):

        return cleaned

    parts = []
    for n in func.body:
        try:
            parts.append(ast.unparse(n))
        except Exception:

            continue
    return "\n".join(parts).strip()
