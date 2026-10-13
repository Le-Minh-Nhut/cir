def test_a_cli_bundle_selection_plans_every_member():
    """Naming a declared bundle on the CLI must plan all of its members.

    The evaluator resolves a bundle id to its members; the acquisition and verification stages must
    agree, or a bundle invocation silently plans no checkpoint at all.
    """
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import load_registry, model_by_id
    from workbench.scripts import pipeline

    registry = load_registry()
    model = model_by_id("limn", registry)
    bundle = model["checkpoint_bundles"][0]
    config = resolve_config()

    for selection, expected in ((bundle["bundle_id"], len(bundle["required_checkpoint_ids"])),
                                (bundle["required_checkpoint_ids"][0], 1)):
        args = pipeline.parser_for().parse_args(
            ["real", "--model", "limn", "--checkpoint", selection, "--evaluate", "--download-checkpoints"])
        stages = pipeline.real_stages(args, config)
        checkpoint = next(stage for stage in stages if stage.name == "checkpoint")
        assert not checkpoint.skipped, selection
        assert len(checkpoint.output_paths) == expected, (selection, checkpoint.output_paths)
        # The acquisition command names the resolved MEMBERS, not the bundle id: the downloader
        # addresses files, so it must be told each member it has to fetch.
        for member in (bundle["required_checkpoint_ids"] if selection == bundle["bundle_id"] else [selection]):
            assert any(member in str(command) for command in checkpoint.commands), (selection, member)
