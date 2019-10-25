def test_solution(capsys):
    import solution  # noqa
    captured = capsys.readouterr()
    actual = captured.out
    with open('solution.txt', 'r') as f:
        expected = f.read()
    assert actual.strip() == expected.strip()
