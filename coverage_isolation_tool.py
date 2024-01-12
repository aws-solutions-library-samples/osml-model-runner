#  Copyright 2024 Amazon.com, Inc. or its affiliates.

import subprocess

TEST_COLUMN_PRINT_WIDTH = 45
FILE_UNDER_TEST_COLUMN_PRINT_WIDTH = 45
COVERAGE_COLUMN_PRINT_WIDTH = 8
MISSING_LINES_COLUMN_PRINT_WIDTH = 13
MISSING_BRANCHES_COLUMN_PRINT_WIDTH = 16

TABLE_WIDTH = (TEST_COLUMN_PRINT_WIDTH + FILE_UNDER_TEST_COLUMN_PRINT_WIDTH + COVERAGE_COLUMN_PRINT_WIDTH +
               MISSING_LINES_COLUMN_PRINT_WIDTH + MISSING_BRANCHES_COLUMN_PRINT_WIDTH + 14)

def pad_string(short_string: str, pad_width: int) -> str:
    if len(short_string) < pad_width:
        return short_string + " " * (pad_width - len(short_string))
    else:
        return short_string


def print_table_break() -> None:
    print("-" * TABLE_WIDTH)

result = subprocess.run(["pytest --co"], shell=True, capture_output=True)
pytest_result_str = result.stdout.decode("utf-8")

test_files = set()
for line in pytest_result_str.split("\n"):
    if "::" in line:
        test_files.add(line.split("::")[0])

exclude = {"test/test_api.py"}

print_table_break()
example_text = "run 'tox -- -s test/aws/osml/model_runner/<test file>' for more details"
table_title = "Isolated Test Coverage"
print(table_title + " " * (TABLE_WIDTH - len(table_title) - len(example_text)) + example_text)
print_table_break()

padded_test_file = pad_string("Test File", TEST_COLUMN_PRINT_WIDTH)
padded_file_under_test = pad_string("File Under Test", FILE_UNDER_TEST_COLUMN_PRINT_WIDTH)
print(f"{padded_test_file} | {padded_file_under_test} | Coverage | Missing lines | Missing branches")
print_table_break()

for test_file in sorted(list(test_files)):
    if test_file not in exclude:

        result = subprocess.run([f"pytest {test_file} --cov aws.osml.model_runner"], shell=True, capture_output=True)
        test_result_str = result.stdout.decode("utf-8")

        expected_file_name = test_file.replace("test/", "").replace("test_", "")
        short_expected_file_name = expected_file_name.replace("aws/osml/model_runner/", "")
        short_test_file_name = test_file.replace("test/aws/osml/model_runner/", "")
        padded_test_name = pad_string(short_test_file_name, TEST_COLUMN_PRINT_WIDTH)
        padded_file_under_test = pad_string(short_expected_file_name, FILE_UNDER_TEST_COLUMN_PRINT_WIDTH)
        matching_coverage_line = None
        for line in test_result_str.split("\n"):
            if expected_file_name in line:
                matching_coverage_line = line
                break
        if matching_coverage_line:
            line_fragments = matching_coverage_line.split()
            coverage = line_fragments[5]
            missing_lines = line_fragments[2]
            missing_brances = line_fragments[3]
            padded_coverage = pad_string(coverage, COVERAGE_COLUMN_PRINT_WIDTH)
            padded_missing_lines = pad_string(missing_lines, MISSING_LINES_COLUMN_PRINT_WIDTH)
            padded_missing_branches = pad_string(missing_brances, MISSING_BRANCHES_COLUMN_PRINT_WIDTH)
            print(f"{padded_test_name} | {padded_file_under_test} | {padded_coverage} | {padded_missing_lines} | "
                  f"{padded_missing_branches} | tox -- -s {test_file}")
        else:
            print(f"Could not find {expected_file_name} in test results")
            print(f"{padded_test_name} | {padded_file_under_test}")
            print(test_result_str)

print_table_break()
