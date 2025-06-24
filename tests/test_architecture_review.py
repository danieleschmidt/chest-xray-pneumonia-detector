import textwrap

from src.architecture_review import parse_module, scan_project, generate_markdown


def test_parse_module(tmp_path):
    module_path = tmp_path / "module.py"
    module_path.write_text(
        textwrap.dedent(
            '''
            """Example module"""
            
            class Foo:
                pass

            def bar():
                return 1
            '''
        )
    )
    info = parse_module(module_path)
    assert info.name == "module"
    assert info.classes == ["Foo"]
    assert info.functions == ["bar"]
    assert info.docstring.strip() == "Example module"


def test_scan_and_generate(tmp_path):
    (tmp_path / "a.py").write_text("def x():\n    pass\n")
    (tmp_path / "b.py").write_text("def y():\n    pass\n")
    modules = scan_project(tmp_path)
    md = generate_markdown(modules)
    assert "# Architecture Overview" in md
    assert "## a" in md and "## b" in md
