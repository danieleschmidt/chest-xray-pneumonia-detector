from importlib.metadata import version, PackageNotFoundError


def main() -> None:
    """Print the installed package version."""
    try:
        pkg_version = version("chest_xray_pneumonia_detector")
    except PackageNotFoundError:  # pragma: no cover - should not happen in package
        pkg_version = "unknown"
    except Exception:  # pragma: no cover - unexpected errors
        # Handle any other unexpected errors gracefully
        pkg_version = "unknown"
    print(pkg_version)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
