import opencc

def convert_simplified_to_traditional(input_file, output_file):
  """Converts a simplified Chinese text file to a traditional Chinese text file.

  Args:
    input_file: The path to the input file.
    output_file: The path to the output file.
  """

  converter = opencc.OpenCC('s2t')

  with open(input_file, "r", encoding="utf-8") as input_file, open(output_file, "w", encoding="utf-8") as output_file:
    for line in input_file:
      converted_line = converter.convert(line)
      output_file.write(converted_line)

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Converts a simplified Chinese text file to a traditional Chinese text file.")
  parser.add_argument("input_file", help="The path to the input file.")
  parser.add_argument("output_file", help="The path to the output file.")
  args = parser.parse_args()

  convert_simplified_to_traditional(args.input_file, args.output_file)