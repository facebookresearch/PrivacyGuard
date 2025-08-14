# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import numpy as np
from privacy_guard.analysis.mia.fdp_analysis_node import (
    FDPAnalysisNode,
    FDPAnalysisNodeOutput,
)
from scipy.stats import norm


class TestFDPAnalysisNode(unittest.TestCase):
    """Test suite for FDPAnalysisNode class."""

    def setUp(self) -> None:
        """Set up test instances with different parameters."""
        # Create analysis nodes with different parameters
        self.default_node = FDPAnalysisNode(m=1000, c=500, c_cap=800)
        self.custom_node = FDPAnalysisNode(
            target_noise=0.5,
            threshold=0.1,
            k=3,
            delta=1e-8,
            m=1000,
            c=500,
            c_cap=800,
        )
        super().setUp()

    def test_initialization_parameters(self) -> None:
        """Test that initialization parameters are correctly set."""
        # Test default parameters
        self.assertIsInstance(self.default_node, FDPAnalysisNode)
        self.assertEqual(self.default_node.threshold, 0.05)
        self.assertEqual(self.default_node.delta, 1e-6)
        self.assertEqual(self.default_node.k, 2)
        self.assertEqual(self.default_node.candidate_noises[1], 0.011)  # 0.001 + 0.01

        # Test custom parameters
        self.assertEqual(self.custom_node.threshold, 0.1)
        self.assertEqual(self.custom_node.delta, 1e-8)
        self.assertEqual(self.custom_node.k, 3)
        self.assertEqual(self.custom_node.candidate_noises[1], 0.51)  # 0.5 + 0.01

        # Test candidate_noises and inverse_blow_up_functions initialization
        self.assertEqual(len(self.default_node.candidate_noises), 1000)
        self.assertEqual(len(self.default_node.inverse_blow_up_functions), 1000)

    def test_gaussianDP_blow_up_function(self) -> None:
        """Test the gaussianDP_blow_up_function static method."""
        # Create a blow-up function with a specific noise level
        noise = 0.5
        blow_up_fn = FDPAnalysisNode.gaussianDP_blow_up_function(noise)

        # Test the function with various inputs
        self.assertAlmostEqual(
            blow_up_fn(0.5), norm.cdf(norm.ppf(0.5) + 1 / 0.5), places=6
        )
        self.assertAlmostEqual(
            blow_up_fn(0.25), norm.cdf(norm.ppf(0.25) + 1 / 0.5), places=6
        )
        self.assertAlmostEqual(
            blow_up_fn(0.75), norm.cdf(norm.ppf(0.75) + 1 / 0.5), places=6
        )

        # Test with different noise level
        noise = 1.0
        blow_up_fn = FDPAnalysisNode.gaussianDP_blow_up_function(noise)
        self.assertAlmostEqual(
            blow_up_fn(0.5), norm.cdf(norm.ppf(0.5) + 1 / 1.0), places=6
        )

    def test_gaussianDP_blow_up_inverse(self) -> None:
        """Test the gaussianDP_blow_up_inverse static method."""
        # Create an inverse blow-up function with a specific noise level
        noise = 0.5
        inverse_blow_up_fn = FDPAnalysisNode.gaussianDP_blow_up_inverse(noise)

        # Test the function with various inputs
        self.assertAlmostEqual(
            inverse_blow_up_fn(0.5), norm.cdf(norm.ppf(0.5) - 1 / 0.5), places=6
        )
        self.assertAlmostEqual(
            inverse_blow_up_fn(0.25), norm.cdf(norm.ppf(0.25) - 1 / 0.5), places=6
        )
        self.assertAlmostEqual(
            inverse_blow_up_fn(0.75), norm.cdf(norm.ppf(0.75) - 1 / 0.5), places=6
        )

        # Test with different noise level
        noise = 1.0
        inverse_blow_up_fn = FDPAnalysisNode.gaussianDP_blow_up_inverse(noise)
        self.assertAlmostEqual(
            inverse_blow_up_fn(0.5), norm.cdf(norm.ppf(0.5) - 1 / 1.0), places=6
        )

    def test_calculate_delta_gaussian(self) -> None:
        """Test the calculate_delta_gaussian method."""
        # Test with specific noise and epsilon values
        noise = 1.0
        epsilon = 1.0
        delta = self.default_node.calculate_delta_gaussian(noise, epsilon)

        # Calculate expected delta manually
        expected_delta = norm.cdf(-epsilon * noise + 1 / (2 * noise)) - np.exp(
            epsilon
        ) * norm.cdf(-epsilon * noise - 1 / (2 * noise))
        self.assertAlmostEqual(delta, expected_delta, places=6)

        # Test with different values
        noise = 2.0
        epsilon = 0.5
        delta = self.default_node.calculate_delta_gaussian(noise, epsilon)
        expected_delta = norm.cdf(-epsilon * noise + 1 / (2 * noise)) - np.exp(
            epsilon
        ) * norm.cdf(-epsilon * noise - 1 / (2 * noise))
        self.assertAlmostEqual(delta, expected_delta, places=6)

    def test_calculate_epsilon_gaussian(self) -> None:
        """Test the calculate_epsilon_gaussian method."""
        # Test with a specific noise value
        noise = 1.0
        epsilon = self.default_node.calculate_epsilon_gaussian(noise)

        # Verify that the calculated epsilon satisfies the delta constraint
        delta = self.default_node.calculate_delta_gaussian(noise, epsilon)
        self.assertLessEqual(delta, self.default_node.delta)

        # Test with a different noise value
        noise = 2.0
        epsilon = self.default_node.calculate_epsilon_gaussian(noise)
        delta = self.default_node.calculate_delta_gaussian(noise, epsilon)
        self.assertLessEqual(delta, self.default_node.delta)

        # Test with custom node (different delta)
        noise = 1.0
        epsilon = self.custom_node.calculate_epsilon_gaussian(noise)
        delta = self.custom_node.calculate_delta_gaussian(noise, epsilon)
        self.assertLessEqual(delta, self.custom_node.delta)

    def test_rh_with_cap(self) -> None:
        """Test the rh_with_cap method."""

        # Create a simple inverse blow-up function for testing
        def test_inverse_blow_up(x: float) -> float:
            return x / 2  # Simple function for testing

        # Test parameters
        alpha = 0.1
        beta = 0.2
        j = 3
        m = 100
        c_cap = 10

        # Call the method
        r, h = self.default_node.rh_with_cap(
            test_inverse_blow_up, alpha, beta, j, m, c_cap
        )

        # Verify the lengths of r and h
        self.assertEqual(len(r), j + 1)
        self.assertEqual(len(h), j + 1)

        # Verify initial values
        self.assertEqual(r[j], alpha)
        self.assertEqual(h[j], beta)

        # Verify the calculation for the remaining values
        for i in range(j - 1, -1, -1):
            self.assertEqual(
                h[i],
                max(
                    h[i + 1], (self.default_node.k - 1) * test_inverse_blow_up(r[i + 1])
                ),
            )
            self.assertEqual(r[i], r[i + 1] + (i / (c_cap - i)) * (h[i] - h[i + 1]))

    def test_audit_rh_with_cap(self) -> None:
        """Test the audit_rh_with_cap method."""

        # Create a simple inverse blow-up function for testing
        def test_inverse_blow_up(x: float) -> float:
            return x / 2  # Simple function that should pass the audit

        # Test parameters
        m = 1000
        c = 100
        c_cap = 200

        # Test with a function that should pass the audit
        result = self.default_node.audit_rh_with_cap(test_inverse_blow_up, m, c, c_cap)
        self.assertTrue(result)

        # Create a function that should fail the audit
        def test_inverse_blow_up_fail(x: float) -> float:
            return x * 2  # Function that should fail the audit

        # Test with a function that should fail the audit
        result = self.default_node.audit_rh_with_cap(
            test_inverse_blow_up_fail, m, c, c_cap
        )
        self.assertFalse(result)

    def test_run_analysis(self) -> None:
        """Test the run_analysis method."""

        # Run the analysis (default args are m=1000, c=500, c_cap=800)
        output = self.default_node.run_analysis()

        # Verify the output type
        self.assertIsInstance(output, FDPAnalysisNodeOutput)

        # Verify the output has the expected attribute
        self.assertTrue(hasattr(output, "eps"))
        self.assertIsInstance(output.eps, float)
        self.assertGreater(output.eps, 0)

        # Test with different parameters
        m = 2000
        c = 1000
        c_cap = 1500

        output = self.default_node.run_analysis_with_parameters(m=m, c=c, c_cap=c_cap)
        self.assertIsInstance(output, FDPAnalysisNodeOutput)
        self.assertGreater(output.eps, 0)

        # Test with custom node
        output = self.custom_node.run_analysis_with_parameters(m=m, c=c, c_cap=c_cap)
        self.assertIsInstance(output, FDPAnalysisNodeOutput)
        self.assertGreater(output.eps, 0)

    def test_run_analysis_assertion(self) -> None:
        """Test that run_analysis raises an assertion error when c > c_cap."""
        # Test parameters where c > c_cap
        m = 1000
        c = 600
        c_cap = 500

        # Verify that an assertion error is raised
        with self.assertRaises(AssertionError):
            self.default_node.run_analysis_with_parameters(m, c, c_cap)

        # Test parameters where c_cap > m
        m = 100
        c = 600
        c_cap = 500

        # Verify that an assertion error is raised
        with self.assertRaises(AssertionError):
            self.default_node.run_analysis_with_parameters(m, c, c_cap)

    def test_output_to_dict(self) -> None:
        """Test that the output can be converted to a dictionary."""
        # Run the analysis
        m = 1000
        c = 500
        c_cap = 800

        output = self.default_node.run_analysis_with_parameters(m=m, c=c, c_cap=c_cap)

        # Convert to dictionary
        output_dict = output.to_dict()

        # Verify the dictionary
        self.assertIsInstance(output_dict, dict)
        self.assertIn("eps", output_dict)
        self.assertEqual(output_dict["eps"], output.eps)


if __name__ == "__main__":
    unittest.main()
