class Predictor:
    def __call__(
        self, input_filenames: str | list[str], output_filenames: str | list[str]
    ):
        assert type(input_filenames) is type(output_filenames)
        if isinstance(input_filenames, str):
            input_filenames = [input_filenames]
            output_filenames = [output_filenames]

        for input_filename, output_filename in zip(
            input_filenames, output_filenames, strict=True
        ):
            with open(input_filename, "rb") as inp, open(output_filename, "wb") as outp:
                outp.write(inp.read())
