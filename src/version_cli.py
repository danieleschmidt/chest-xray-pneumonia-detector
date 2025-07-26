from importlib.metadata import version, PackageNotFoundError


def main() -> None:
    """Print the installed package version to stdout.

    Retrieves and displays the version of the chest_xray_pneumonia_detector
    package using importlib.metadata. Handles package discovery errors
    gracefully by displaying 'unknown' for missing or corrupted installations.

    Returns
    -------
    None
        Prints version string to stdout and exits.

    Notes
    -----
    - Primary entry point for the cxr-version CLI command
    - Uses importlib.metadata for reliable version detection
    - Fallback to 'unknown' prevents CLI crashes from version errors
    - Designed for CI/CD pipelines and debugging workflows

    Examples
    --------
    $ cxr-version
    0.2.0
    """
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
