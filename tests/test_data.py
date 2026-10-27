from src.data import make_demo_data, validate_data

def test_demo_data_has_target_and_both_classes():
    df = make_demo_data(n=500)
    checked = validate_data(df)
    assert "Revenue" in checked.columns
    assert set(checked["Revenue"].unique()) == {0, 1}

def test_validation_removes_duplicates():
    df = make_demo_data(n=500)
    doubled = __import__("pandas").concat([df, df.iloc[[0]]], ignore_index=True)
    assert len(validate_data(doubled)) <= len(doubled)
