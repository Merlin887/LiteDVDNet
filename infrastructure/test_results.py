from dataclasses import dataclass, field

from future.moves import itertools
from tabulate import tabulate


@dataclass
class TestCaseResult:
    model_name: str
    test_case_name: str
    noise_level: int
    psnr_noisy: float = field(default=0.0)
    psnr_clean: float = field(default=0.0)
    strred_score: float = field(default=0.0)
    ssim_score: float = field(default=0.0)
    total_denoising_time: float = field(default=0.0)
    single_frame_time: float = field(default=0.0)

@dataclass
class TestResult:
    test_name: str
    noise_level: float = field(default=0.0)
    psnr_clean: float = field(default=0.0)
    strred_score: float = field(default=0.0)
    ssim_score: float = field(default=0.0)
    single_frame_time: float = field(default=0.0)

@dataclass
class ModelResult:
    model_name: str
    test_results: list[TestResult]
    mean_psnr: float = field(default=0.0)
    mean_sttred: float = field(default=0.0)
    mean_ssim: float = field(default=0.0)
    mean_frame_time: float = field(default=0.0)

@dataclass
class TestSuiteResult:
    model_results: list[ModelResult]

    def get_results(self) -> str:

        headers = ["Model", "Noise Level", "Mean PSNR", "Mean STTRED", "Mean SSIM", "Denoise Time (msec)"]
        table = []

        # Find the best scores
        best_psnr = max(model_result.mean_psnr for model_result in self.model_results)
        best_sttred = max(model_result.mean_sttred for model_result in self.model_results)
        best_ssim = min(model_result.mean_ssim for model_result in self.model_results)
        best_frame_time = min(model_result.mean_frame_time for model_result in self.model_results)

        for model_result in self.model_results:
            row = [
                model_result.model_name,
                model_result.test_results[0].noise_level,
                f"{'Best ' if model_result.mean_psnr == best_psnr else ''}{model_result.mean_psnr:.2f}",
                f"{'Best ' if model_result.mean_sttred == best_sttred else ''}{model_result.mean_sttred:.2f}",
                f"{'Best ' if model_result.mean_ssim == best_ssim else ''}{model_result.mean_ssim:.2f}",
                f"{'Best ' if model_result.mean_frame_time == best_frame_time else ''}{model_result.mean_frame_time:.2f}",
            ]

            table.append(row)


        return tabulate(table, headers=headers, tablefmt="grid")


def organize_results(test_case_results: list[TestCaseResult]) -> TestSuiteResult:

    sorted_results = sorted(test_case_results, key=lambda x: x.model_name)
    grouped_results = itertools.groupby(sorted_results, key=lambda x: x.model_name)

    model_results = []

    for model_name, group in grouped_results:

        test_case_results = list(group)
        mean_psnr = sum(result.psnr_clean for result in test_case_results) / len(test_case_results)
        mean_sttred = sum(result.strred_score for result in test_case_results) / len(test_case_results)
        mean_ssim = sum(result.ssim_score for result in test_case_results) / len(test_case_results)
        mean_frame_time = sum(result.single_frame_time for result in test_case_results) / len(test_case_results)

        model_result = ModelResult(
            model_name=model_name,
            test_results=convert_to_test_results(test_case_results),
            mean_psnr=mean_psnr,
            mean_sttred=mean_sttred,
            mean_ssim=mean_ssim,
            mean_frame_time=mean_frame_time
        )

        model_results.append(model_result)

    model_results = sorted(model_results, key=lambda x: x.mean_psnr)

    return TestSuiteResult(model_results)

def convert_to_test_results(test_case_results: list[TestCaseResult]) -> list[TestResult]:
    test_results = []

    for test_case_result in test_case_results:
        test_result = TestResult(
            test_name=test_case_result.test_case_name,
            noise_level=test_case_result.noise_level,
            psnr_clean=test_case_result.psnr_clean,
            strred_score=test_case_result.strred_score,
            ssim_score=test_case_result.ssim_score,
            single_frame_time=test_case_result.single_frame_time
        )
        test_results.append(test_result)

    return test_results

def get_test_case_table(test_case_results: list[TestCaseResult]):
    sorted_results = sorted(test_case_results, key=lambda x: x.test_case_name)
    grouped_results = itertools.groupby(sorted_results, key=lambda x: x.test_case_name)

    headers = ["Model", "Dataset", "Noise Level", "PSNR", "STTRED", "SSIM", "Denoise Time (msec)"]
    table = []

    for test_case_name, group in grouped_results:
        test_results = list(group)
        test_results = sorted(test_results, key=lambda x: x.psnr_clean)

        for result in test_results:
            row = [
                result.model_name,
                result.test_case_name,
                result.noise_level,
                f"{result.psnr_clean:.2f}",
                f"{result.strred_score:.2f}",
                f"{result.ssim_score:.2f}",
                f"{result.single_frame_time:.2f}",
            ]

            table.append(row)

    return tabulate(table, headers=headers, tablefmt="grid")