from ultralytics import YOLO
from ultralytics.engine.results import Results


class Predictor:
    def __init__(self, model_to_weights_path: dict[str, str]):
        self.available_models: dict[str, YOLO] = {}
        for model_name, weights_path in model_to_weights_path.items():
            self.available_models[model_name] = YOLO(weights_path)

    def __call__(
        self,
        input_filenames: str | list[str],
        output_filenames: str | list[str],
        *,
        model: str,
        classes: list[int],
    ) -> Results | list[Results]:
        assert type(input_filenames) is type(output_filenames)
        if isinstance(input_filenames, str):
            input_filenames = [input_filenames]
            output_filenames = [output_filenames]

        results: list[Results] = self.available_models[model](
            input_filenames, classes=classes
        )

        for result, output_filename in zip(results, output_filenames, strict=True):
            result.show()

            result.save(filename=output_filename)
        if len(results) == 1:
            return results[0]
        return results
