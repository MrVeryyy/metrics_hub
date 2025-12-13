def test_import_main_and_metrics():
    """
    Very lightweight import test.
    This should never touch heavy dependencies (torch / lpips).
    """
    import main
    import metrics

    assert hasattr(main, "__file__")
    assert hasattr(metrics, "__file__")


def test_registry_has_metrics():
    """
    Registry smoke test.
    We only check names, not instantiation of heavy models.
    """
    import main  # trigger side-effect imports if any
    from main.registry import list_metrics

    ms = list_metrics()
    assert isinstance(ms, list)
    assert len(ms) > 0, "registry is empty; metrics may not be registered"

    # Check expected metric names (do NOT instantiate them)
    assert "psnr" in ms
    assert "ssim" in ms
