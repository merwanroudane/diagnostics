import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.api as sm
from PIL import Image
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="OLS Assumptions Violations Guide",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
        text-align: center;
    }
    .sub-header {
        font-size: 30px;
        font-weight: bold;
        color: #1F2937;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #374151;
        margin-top: 25px;
        margin-bottom: 10px;
    }
    .highlight {
        background-color: #FBBF24;
        padding: 0 4px;
        border-radius: 4px;
    }
    .important-note {
        background-color: #FEF3C7;
        padding: 15px;
        border-left: 5px solid #F59E0B;
        border-radius: 4px;
        margin: 20px 0;
    }
    .card {
        background-color: #F9FAFB;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .caption {
        font-size: 14px;
        color: #6B7280;
        text-align: center;
        margin-top: 10px;
    }
    .formula {
        background-color: #E0E7FF;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<div class="main-header">Understanding OLS Assumption Violations by Dr Merwan Roudane </div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card">This interactive guide helps you teach students about OLS assumption violations, with special focus on how these violations affect the distributions of OLS estimators, test statistics (F and t), and standard errors. The goal is to clearly demonstrate why these problems matter in practice and how they impact statistical inference.</div>',
    unsafe_allow_html=True)

# Navigation sidebar
with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Select a topic:",
        ["Introduction to OLS Assumptions",
         "Linearity Violation",
         "Normality Violation",
         "Homoscedasticity Violation",
         "Autocorrelation Violation",
         "Multicollinearity",
         "Endogeneity",
         "Comparative Analysis",
         "Practical Examples",
         "Summary and Quiz"]
    )

    st.markdown("---")
    st.markdown("### Teaching Tips")
    st.info("""
    - Use the interactive simulations to demonstrate concepts visually
    - Encourage students to change parameters and observe effects
    - Connect theoretical implications to real-world research problems
    - Emphasize practical significance over mathematical details
    """)


# Helper functions for generating data and visualizations
def generate_linear_data(n=100, beta0=2, beta1=3, error_sd=1, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(0, 10, n)
    epsilon = np.random.normal(0, error_sd, n)
    y = beta0 + beta1 * x + epsilon
    return x, y, epsilon


def generate_nonlinear_data(n=100, beta0=2, beta1=3, beta2=0.5, error_sd=1, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(0, 10, n)
    epsilon = np.random.normal(0, error_sd, n)
    y = beta0 + beta1 * x + beta2 * x ** 2 + epsilon
    return x, y, epsilon


def generate_heteroscedastic_data(n=100, beta0=2, beta1=3, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(0, 10, n)
    # Error variance increases with x
    epsilon = np.random.normal(0, 0.5 + 0.5 * x, n)
    y = beta0 + beta1 * x + epsilon
    return x, y, epsilon


def generate_autocorrelated_data(n=100, beta0=2, beta1=3, rho=0.7, error_sd=1, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(0, 10, n)

    # Generate autocorrelated errors
    epsilon = np.zeros(n)
    epsilon[0] = np.random.normal(0, error_sd)
    for i in range(1, n):
        epsilon[i] = rho * epsilon[i - 1] + np.random.normal(0, error_sd)

    y = beta0 + beta1 * x + epsilon
    return x, y, epsilon


def generate_multicollinear_data(n=100, beta0=2, beta1=3, beta2=1.5, correlation=0.9, error_sd=1, seed=42):
    np.random.seed(seed)
    x1 = np.random.uniform(0, 10, n)

    # Generate x2 that is correlated with x1
    x2 = correlation * x1 + (1 - correlation) * np.random.uniform(0, 10, n)

    epsilon = np.random.normal(0, error_sd, n)
    y = beta0 + beta1 * x1 + beta2 * x2 + epsilon

    return x1, x2, y, epsilon


def generate_endogenous_data(n=100, beta0=2, beta1=3, endogeneity=0.7, error_sd=1, seed=42):
    np.random.seed(seed)

    # Generate unobserved variable that affects both x and y
    unobserved = np.random.normal(0, 1, n)

    # x is correlated with the unobserved variable
    x = np.random.uniform(0, 10, n) + endogeneity * unobserved

    # Error term is also affected by the unobserved variable
    epsilon = np.random.normal(0, error_sd, n) + endogeneity * unobserved

    # Generate y
    y = beta0 + beta1 * x + epsilon

    return x, y, epsilon, unobserved


def plot_residuals(x, residuals, title="Residuals Plot"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='-')
    ax.set_xlabel("X")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig


def plot_qq(residuals):
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q Plot")
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig


def compare_distributions(normal_sample, affected_sample, title="Comparison of Sampling Distributions"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(normal_sample, ax=ax, label="Under correct assumptions", color="blue")
    sns.kdeplot(affected_sample, ax=ax, label="Under violated assumptions", color="red")
    ax.axvline(np.mean(normal_sample), color='blue', linestyle='--', alpha=0.7)
    ax.axvline(np.mean(affected_sample), color='red', linestyle='--', alpha=0.7)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig


def run_monte_carlo(generate_data_func, n_simulations=1000, n_samples=100, **kwargs):
    betas = []
    t_stats = []
    std_errors = []

    for i in range(n_simulations):
        if generate_data_func.__name__ == 'generate_multicollinear_data':
            x1, x2, y, _ = generate_data_func(n=n_samples, seed=i, **kwargs)
            X = sm.add_constant(np.column_stack((x1, x2)))
            model = sm.OLS(y, X).fit()
            betas.append([model.params[1], model.params[2]])  # Beta1 and Beta2
            t_stats.append([model.tvalues[1], model.tvalues[2]])
            std_errors.append([model.bse[1], model.bse[2]])
        else:
            x, y, _ = generate_data_func(n=n_samples, seed=i, **kwargs)
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            betas.append(model.params[1])  # Beta1
            t_stats.append(model.tvalues[1])
            std_errors.append(model.bse[1])

    return np.array(betas), np.array(t_stats), np.array(std_errors)


# Introduction to OLS Assumptions
if page == "Introduction to OLS Assumptions":
    st.markdown('<div class="sub-header">Introduction to OLS Assumptions</div>', unsafe_allow_html=True)

    st.markdown("""
    The Ordinary Least Squares (OLS) method is the foundation of linear regression analysis. Its power comes from a set of assumptions that, when satisfied, give the OLS estimators desirable statistical properties. These properties are what make hypothesis testing and confidence intervals reliable.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">The Key OLS Assumptions</div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Linearity**: The relationship between independent and dependent variables is linear
        2. **Independence**: Observations are independent of each other
        3. **Normality**: Error terms are normally distributed
        4. **Homoscedasticity**: Error terms have constant variance
        5. **No Multicollinearity**: Independent variables are not perfectly correlated
        6. **No Endogeneity**: Independent variables are not correlated with the error term
        """)

        st.markdown('<div class="section-header">Why These Assumptions Matter</div>', unsafe_allow_html=True)
        st.markdown("""
        When all assumptions are met, OLS estimators have these important properties:

        - **Unbiasedness**: The expected value of the estimator equals the true parameter value
        - **Efficiency**: They have the smallest variance among all unbiased linear estimators (BLUE)
        - **Consistency**: As sample size increases, estimators converge to the true values
        - **Normal Sampling Distribution**: Enables reliable hypothesis testing and confidence intervals
        """)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### The Linear Model")
        st.markdown(r'''
        $$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_k X_k + \varepsilon$$

        Where:
        - $Y$ is the dependent variable
        - $X_i$ are independent variables
        - $\beta_i$ are the coefficients
        - $\varepsilon$ is the error term
        ''')
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="important-note">', unsafe_allow_html=True)
        st.markdown("""
        **Key Point for Students**: 

        Violations of OLS assumptions don't just affect theoretical properties—they directly impact your ability to draw valid conclusions from your data. Understanding these violations helps you avoid making incorrect inferences in your research.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">What Happens When Assumptions Are Violated?</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Biased Estimators")
        st.markdown("""
        - Coefficient estimates systematically deviate from true values
        - Leads to incorrect conclusions about relationships
        - May suggest relationships that don't exist or miss ones that do
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Incorrect Standard Errors")
        st.markdown("""
        - Standard errors become unreliable
        - Confidence intervals are too narrow or too wide
        - Can't properly assess the precision of estimates
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Invalid Hypothesis Tests")
        st.markdown("""
        - t-statistics and F-statistics become unreliable
        - p-values are misleading
        - Type I and II errors occur more frequently
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # Add a demonstration
    st.markdown('<div class="section-header">Visual Demonstration: OLS Ideal Case</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    n_samples = st.slider("Number of observations", min_value=20, max_value=200, value=100, step=10,
                          key="intro_n_samples")
    error_sd = st.slider("Error term standard deviation", min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                         key="intro_error_sd")

    x, y, epsilon = generate_linear_data(n=n_samples, error_sd=error_sd)
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, alpha=0.6, label='Data points')
        ax.plot(x, model.predict(), color='red', label='OLS regression line')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Linear Regression under Ideal Conditions")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Regression Results")
        st.markdown(f"""
        - Estimate for β₀ (intercept): {model.params[0]:.4f}
        - Estimate for β₁ (slope): {model.params[1]:.4f}
        - Standard error for β₁: {model.bse[1]:.4f}
        - t-statistic for β₁: {model.tvalues[1]:.4f}
        - p-value for β₁: {model.pvalues[1]:.4f}
        - R-squared: {model.rsquared:.4f}
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        # Q-Q plot
        fig = plot_qq(model.resid)
        st.pyplot(fig)
        st.markdown(
            '<div class="caption">Q-Q plot of residuals. Points along the line indicate normally distributed errors.</div>',
            unsafe_allow_html=True)

# Linearity Violation
elif page == "Linearity Violation":
    st.markdown('<div class="sub-header">Violation of Linearity Assumption</div>', unsafe_allow_html=True)

    st.markdown("""
    The linearity assumption states that the relationship between the dependent variable and each independent variable is linear. When this assumption is violated, OLS attempts to fit a straight line to a non-linear relationship.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Effects on OLS Estimators</div>', unsafe_allow_html=True)
        st.markdown("""
        When the true relationship is non-linear, but we fit a linear model:

        1. **Biased Estimators**: Coefficients are systematically biased, not capturing the true relationship
        2. **Systematic Patterns in Residuals**: Residuals will show patterns rather than random scatter
        3. **Underfitting**: The model fails to capture the true complexity of the relationship
        """)

        st.markdown('<div class="section-header">Impact on Statistical Inference</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Distribution of Estimators**: No longer centered around true parameter values
        - **t-statistics and F-statistics**: Become unreliable due to model misspecification
        - **Confidence Intervals**: Don't have the correct coverage probability
        - **R-squared**: Artificially low, underestimating the explanatory power of variables
        """)

    with col2:
        st.markdown('<div class="important-note">', unsafe_allow_html=True)
        st.markdown("""
        **Why Students Should Care**:

        Linearity violations can lead to completely incorrect conclusions about relationships in your data. You might:

        - Miss important relationships
        - Underestimate effect sizes
        - Make incorrect predictions
        - Fail to detect significant effects
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Mathematical Expression")
        st.markdown(r'''
        True model: $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \varepsilon$

        Misspecified model: $Y = \beta_0 + \beta_1 X + \varepsilon$

        This misspecification leads to bias in $\hat{\beta}_1$ as it tries to account for both linear and quadratic effects.
        ''')
        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive demonstration
    st.markdown('<div class="section-header">Interactive Demonstration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        degree_of_nonlinearity = st.slider("Degree of nonlinearity (β₂)",
                                           min_value=0.0, max_value=1.0, value=0.3, step=0.1,
                                           key="nonlinearity_slider")

    with col2:
        n_samples = st.slider("Number of observations",
                              min_value=30, max_value=200, value=100, step=10,
                              key="linearity_n_slider")

    # Generate nonlinear data
    x, y, epsilon = generate_nonlinear_data(n=n_samples, beta2=degree_of_nonlinearity)

    # Fit linear model (misspecified)
    X_linear = sm.add_constant(x)
    model_linear = sm.OLS(y, X_linear).fit()

    # Fit correct nonlinear model
    X_nonlinear = sm.add_constant(np.column_stack((x, x ** 2)))
    model_nonlinear = sm.OLS(y, X_nonlinear).fit()

    col1, col2 = st.columns([1, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, alpha=0.6, label='Data points')

        # Sort for smooth line plotting
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]

        # Plot linear fit
        ax.plot(x_sorted, model_linear.predict()[sort_idx], color='red', label='Linear fit (misspecified)')

        # Plot nonlinear fit
        ax.plot(x_sorted, model_nonlinear.predict()[sort_idx], color='green', label='Quadratic fit (correct)')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Linear vs. Nonlinear Relationship")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        st.markdown(
            '<div class="caption">As the nonlinearity increases, the misspecification of the linear model becomes more apparent.</div>',
            unsafe_allow_html=True)

    with col2:
        # Plot residuals from linear model
        fig = plot_residuals(x, model_linear.resid, "Residuals from Linear Model")
        st.pyplot(fig)
        st.markdown(
            '<div class="caption">Notice the U-shaped pattern in residuals, indicating model misspecification.</div>',
            unsafe_allow_html=True)

    # Monte Carlo simulation to show impact on estimator distribution
    st.markdown('<div class="section-header">Monte Carlo Simulation: Impact on Estimator Distribution</div>',
                unsafe_allow_html=True)

    if st.button("Run Monte Carlo Simulation for Linearity Violation", key="linearity_mc_button"):
        with st.spinner("Running simulation..."):
            # Simulation for correctly specified model (true linear relationship)
            betas_correct, t_stats_correct, se_correct = run_monte_carlo(
                generate_linear_data, n_simulations=1000, n_samples=n_samples
            )

            # Simulation for misspecified model (true nonlinear relationship, but using linear model)
            betas_incorrect, t_stats_incorrect, se_incorrect = run_monte_carlo(
                generate_nonlinear_data, n_simulations=1000, n_samples=n_samples, beta2=degree_of_nonlinearity
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                # Compare beta distributions
                fig = compare_distributions(betas_correct, betas_incorrect,
                                            "Distribution of β₁ Estimates")
                st.pyplot(fig)
                st.markdown('<div class="caption">The misspecified model leads to biased estimates of β₁.</div>',
                            unsafe_allow_html=True)

            with col2:
                # Compare t-statistic distributions
                fig = compare_distributions(t_stats_correct, t_stats_incorrect,
                                            "Distribution of t-statistics for β₁")
                st.pyplot(fig)
                st.markdown(
                    '<div class="caption">t-statistics from the misspecified model don\'t follow the expected t-distribution.</div>',
                    unsafe_allow_html=True)

            st.markdown('<div class="important-note">', unsafe_allow_html=True)
            st.markdown(f"""
            **Key Findings**:

            1. **Bias in coefficient estimates**: 
               - True β₁ = 3.0
               - Average estimate under correct specification: {np.mean(betas_correct):.4f}
               - Average estimate under misspecification: {np.mean(betas_incorrect):.4f}
               - Bias: {np.mean(betas_incorrect) - 3.0:.4f}

            2. **Impact on hypothesis testing**:
               - Rejection rate of H₀: β₁ = 0 (at 5% level):
               - Correct model: {np.mean(np.abs(t_stats_correct) > 1.96) * 100:.1f}%
               - Misspecified model: {np.mean(np.abs(t_stats_incorrect) > 1.96) * 100:.1f}%

            3. **Impact on standard errors**:
               - Average SE under correct specification: {np.mean(se_correct):.4f}
               - Average SE under misspecification: {np.mean(se_incorrect):.4f}
            """)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Detection and Solutions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### How to Detect Linearity Violations")
        st.markdown("""
        1. **Residual plots**: Look for patterns (U-shaped or inverted U)
        2. **Partial regression plots**: Examine relationships between each predictor and the outcome
        3. **Added variable plots**: Check if adding quadratic or other terms improves fit
        4. **RESET test**: Formal statistical test for functional form misspecification
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Solutions to Linearity Violations")
        st.markdown("""
        1. **Transformation of variables**: Log, square root, or Box-Cox transformations
        2. **Polynomial terms**: Adding squared or cubic terms
        3. **Splines or piecewise regression**: For more complex nonlinear relationships
        4. **Non-parametric methods**: When the functional form is unknown
        5. **Generalized Additive Models (GAMs)**: Allow flexible functional forms
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Normality Violation
elif page == "Normality Violation":
    st.markdown('<div class="sub-header">Violation of Normality Assumption</div>', unsafe_allow_html=True)

    st.markdown("""
    The normality assumption states that the error terms in the regression model are normally distributed. 
    This assumption is particularly important for hypothesis testing and confidence interval construction.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Effects on OLS Estimators</div>', unsafe_allow_html=True)
        st.markdown("""
        When error terms are not normally distributed:

        1. **Unbiasedness Preserved**: OLS estimators remain unbiased (if other assumptions hold)
        2. **Efficiency**: OLS may no longer be the most efficient estimator
        3. **Central Limit Theorem**: With large samples, estimators are approximately normally distributed regardless
        """)

        st.markdown('<div class="section-header">Impact on Statistical Inference</div>', unsafe_allow_html=True)
        st.markdown("""
        - **t-statistics and F-statistics**: No longer follow t and F distributions in small samples
        - **Confidence Intervals**: May not have the correct coverage probability
        - **Hypothesis Tests**: p-values become unreliable, especially in small samples
        - **Type I and II Errors**: Probability of making these errors changes
        """)

    with col2:
        st.markdown('<div class="important-note">', unsafe_allow_html=True)
        st.markdown("""
        **Why Students Should Care**:

        Normality violations affect:

        - Validity of hypothesis tests
        - Reliability of confidence intervals
        - Predictions for extreme values
        - Outlier detection

        The impact is most severe with:
        - Small sample sizes
        - Heavy-tailed distributions
        - Highly skewed error distributions
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### The Central Limit Theorem")
        st.markdown(r'''
        As sample size increases:

        $$\sqrt{n}(\hat{\beta} - \beta) \xrightarrow{d} N(0, \sigma^2(X'X)^{-1})$$

        This means normality matters less for large samples, but is crucial for small samples.
        ''')
        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive demonstration
    st.markdown('<div class="section-header">Interactive Demonstration</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        error_distribution = st.selectbox(
            "Error Distribution",
            ["Normal", "T with 3 df (heavy-tailed)", "Chi-Square (skewed)", "Uniform (light-tailed)"],
            key="normality_dist_select"
        )

    with col2:
        n_samples = st.slider(
            "Sample Size",
            min_value=10, max_value=500, value=50, step=10,
            key="normality_n_slider"
        )

    with col3:
        beta1_true = st.slider(
            "True β₁ Value",
            min_value=0.0, max_value=5.0, value=3.0, step=0.5,
            key="normality_beta_slider"
        )

    # Generate data with different error distributions
    np.random.seed(42)
    x = np.random.uniform(0, 10, n_samples)

    if error_distribution == "Normal":
        epsilon = np.random.normal(0, 1, n_samples)
    elif error_distribution == "T with 3 df (heavy-tailed)":
        epsilon = np.random.standard_t(3, n_samples)
    elif error_distribution == "Chi-Square (skewed)":
        epsilon = np.random.chisquare(3, n_samples) - 3  # Mean-centered
    else:  # Uniform
        epsilon = np.random.uniform(-np.sqrt(3), np.sqrt(3), n_samples)  # Same variance as N(0,1)

    y = 2 + beta1_true * x + epsilon

    # Fit OLS model
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    col1, col2 = st.columns([1, 1])

    with col1:
        # Plot regression with error distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, alpha=0.6, label='Data points')
        ax.plot(np.sort(x), model.predict()[np.argsort(x)], color='red', label='OLS regression line')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Regression with {error_distribution} Errors")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Display error distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(epsilon, kde=True, ax=ax)
        ax.set_title(f"Distribution of Error Terms: {error_distribution}")
        ax.set_xlabel("Error Value")
        ax.axvline(x=0, color='red', linestyle='--')
        st.pyplot(fig)

    with col2:
        # Q-Q plot
        fig = plot_qq(model.resid)
        st.pyplot(fig)
        st.markdown('<div class="caption">Q-Q plot shows deviations from normality in the error terms.</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Regression Results")
        st.markdown(f"""
        - True β₁: {beta1_true:.4f}
        - Estimated β₁: {model.params[1]:.4f}
        - Standard error: {model.bse[1]:.4f}
        - t-statistic: {model.tvalues[1]:.4f}
        - p-value: {model.pvalues[1]:.4f}
        - 95% Confidence Interval: [{model.conf_int().iloc[1, 0]:.4f}, {model.conf_int().iloc[1, 1]:.4f}]
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # Monte Carlo simulation to show impact on hypothesis testing
    st.markdown('<div class="section-header">Monte Carlo Simulation: Impact on Hypothesis Testing</div>',
                unsafe_allow_html=True)

    if st.button("Run Monte Carlo Simulation for Normality Violation", key="normality_mc_button"):
        with st.spinner("Running simulation..."):
            # Functions to generate data with different error distributions
            def generate_normal_data(n=100, beta0=2, beta1=3, error_sd=1, seed=42):
                np.random.seed(seed)
                x = np.random.uniform(0, 10, n)
                epsilon = np.random.normal(0, error_sd, n)
                y = beta0 + beta1 * x + epsilon
                return x, y, epsilon


            def generate_t3_data(n=100, beta0=2, beta1=3, error_sd=1, seed=42):
                np.random.seed(seed)
                x = np.random.uniform(0, 10, n)
                epsilon = np.random.standard_t(3, n) * error_sd
                y = beta0 + beta1 * x + epsilon
                return x, y, epsilon


            def generate_chisq_data(n=100, beta0=2, beta1=3, error_sd=1, seed=42):
                np.random.seed(seed)
                x = np.random.uniform(0, 10, n)
                epsilon = (np.random.chisquare(3, n) - 3) * error_sd / np.sqrt(6)  # Same variance as N(0,1)
                y = beta0 + beta1 * x + epsilon
                return x, y, epsilon


            # Run simulations for different error distributions
            betas_normal, t_stats_normal, se_normal = run_monte_carlo(
                generate_normal_data, n_simulations=1000, n_samples=n_samples, beta1=beta1_true
            )

            if error_distribution == "T with 3 df (heavy-tailed)":
                betas_nonnormal, t_stats_nonnormal, se_nonnormal = run_monte_carlo(
                    generate_t3_data, n_simulations=1000, n_samples=n_samples, beta1=beta1_true
                )
                dist_name = "t(3)"
            elif error_distribution == "Chi-Square (skewed)":
                betas_nonnormal, t_stats_nonnormal, se_nonnormal = run_monte_carlo(
                    generate_chisq_data, n_simulations=1000, n_samples=n_samples, beta1=beta1_true
                )
                dist_name = "Chi-Square"
            else:  # Keep normal for uniform case
                betas_nonnormal, t_stats_nonnormal, se_nonnormal = betas_normal, t_stats_normal, se_normal
                dist_name = "Uniform"

            col1, col2 = st.columns([1, 1])

            with col1:
                # Compare sampling distributions of beta
                fig = compare_distributions(betas_normal, betas_nonnormal,
                                            f"Sampling Distribution of β₁ (n={n_samples})")
                st.pyplot(fig)
                st.markdown(
                    '<div class="caption">Sampling distribution of β₁ under normal errors vs. non-normal errors.</div>',
                    unsafe_allow_html=True)

            with col2:
                # Compare sampling distributions of t-statistics
                fig = compare_distributions(t_stats_normal, t_stats_nonnormal,
                                            f"Sampling Distribution of t-statistics (n={n_samples})")
                st.pyplot(fig)
                st.markdown(
                    '<div class="caption">Distribution of t-statistics under normal errors vs. non-normal errors.</div>',
                    unsafe_allow_html=True)

            # Calculate rejection rates
            alpha_levels = [0.01, 0.05, 0.10]
            rejection_rates_normal = [np.mean(np.abs(t_stats_normal) > stats.t.ppf(1 - alpha / 2, n_samples - 2)) for
                                      alpha in alpha_levels]
            rejection_rates_nonnormal = [np.mean(np.abs(t_stats_nonnormal) > stats.t.ppf(1 - alpha / 2, n_samples - 2))
                                         for alpha in alpha_levels]

            st.markdown('<div class="important-note">', unsafe_allow_html=True)
            st.markdown(f"""
            **Key Findings**:

            1. **Impact on coefficient estimates**:
               - True β₁ = {beta1_true:.1f}
               - Average β₁ (normal errors): {np.mean(betas_normal):.4f}
               - Average β₁ ({dist_name} errors): {np.mean(betas_nonnormal):.4f}
               - Standard deviation of β₁ (normal): {np.std(betas_normal):.4f}
               - Standard deviation of β₁ ({dist_name}): {np.std(betas_nonnormal):.4f}

            2. **Impact on hypothesis testing (H₀: β₁ = 0)**:
               - Nominal vs. actual rejection rates:

               | Alpha Level | Normal Errors | {dist_name} Errors | Difference |
               |------------|--------------|-----------------|------------|
               | 1%         | {rejection_rates_normal[0] * 100:.1f}% | {rejection_rates_nonnormal[0] * 100:.1f}% | {(rejection_rates_nonnormal[0] - rejection_rates_normal[0]) * 100:.1f}% |
               | 5%         | {rejection_rates_normal[1] * 100:.1f}% | {rejection_rates_nonnormal[1] * 100:.1f}% | {(rejection_rates_nonnormal[1] - rejection_rates_normal[1]) * 100:.1f}% |
               | 10%        | {rejection_rates_normal[2] * 100:.1f}% | {rejection_rates_nonnormal[2] * 100:.1f}% | {(rejection_rates_nonnormal[2] - rejection_rates_normal[2]) * 100:.1f}% |

            3. **Confidence Interval Width**:
               - Average 95% CI width (normal): {np.mean(1.96 * 2 * se_normal):.4f}
               - Average 95% CI width ({dist_name}): {np.mean(1.96 * 2 * se_nonnormal):.4f}
            """)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Detection and Solutions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### How to Detect Normality Violations")
        st.markdown("""
        1. **Histogram of residuals**: Check for symmetry, skewness, kurtosis
        2. **Q-Q plots**: Compare residuals to theoretical normal distribution
        3. **Formal tests**:
           - Shapiro-Wilk test
           - Jarque-Bera test
           - Kolmogorov-Smirnov test
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Solutions to Normality Violations")
        st.markdown("""
        1. **Increase sample size**: Rely on Central Limit Theorem
        2. **Transform dependent variable**: Log, square root, Box-Cox
        3. **Robust regression**: Less sensitive to extreme observations
        4. **Bootstrapping**: Estimate sampling distributions without normality assumption
        5. **Quantile regression**: Focus on conditional medians instead of means
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Homoscedasticity Violation
elif page == "Homoscedasticity Violation":
    st.markdown('<div class="sub-header">Violation of Homoscedasticity Assumption</div>', unsafe_allow_html=True)

    st.markdown("""
    The homoscedasticity assumption requires that the error terms have constant variance across all levels of the independent variables. 
    When this assumption is violated, we have heteroscedasticity—variance of errors changes with the values of predictors.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Effects on OLS Estimators</div>', unsafe_allow_html=True)
        st.markdown("""
        When errors have non-constant variance:

        1. **Unbiasedness Preserved**: OLS estimators remain unbiased (if other assumptions hold)
        2. **Efficiency Lost**: OLS is no longer BLUE (Best Linear Unbiased Estimator)
        3. **Incorrect Standard Errors**: Standard errors are biased, usually underestimated
        """)

        st.markdown('<div class="section-header">Impact on Statistical Inference</div>', unsafe_allow_html=True)
        st.markdown("""
        - **t-statistics**: Calculated using incorrect standard errors, leading to wrong conclusions
        - **Confidence Intervals**: Too narrow, giving false confidence in estimates
        - **Hypothesis Tests**: Inflated Type I error rates (rejecting true null hypotheses too often)
        - **F-tests**: Also affected by incorrect variance estimates
        """)

    with col2:
        st.markdown('<div class="important-note">', unsafe_allow_html=True)
        st.markdown("""
        **Why Students Should Care**:

        Heteroscedasticity leads to:

        - Misleadingly significant results
        - Underestimated standard errors
        - False confidence in findings
        - Invalid prediction intervals

        Common in:
        - Cross-sectional data
        - Financial time series
        - Data with large range of predictor values
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Mathematical Expression")
        st.markdown(r'''
        Under heteroscedasticity:

        $$Var(\varepsilon_i|X_i) = \sigma_i^2$$

        OLS assumes: $Var(\varepsilon_i|X_i) = \sigma^2$ (constant)

        The variance of $\hat{\beta}$ becomes:

        $$Var(\hat{\beta}) = (X'X)^{-1}X'\Omega X(X'X)^{-1}$$

        Where $\Omega = diag(\sigma_1^2, \sigma_2^2, ..., \sigma_n^2)$
        ''')
        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive demonstration
    st.markdown('<div class="section-header">Interactive Demonstration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        hetero_strength = st.slider(
            "Degree of Heteroscedasticity",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            key="hetero_strength_slider"
        )

    with col2:
        n_samples = st.slider(
            "Sample Size",
            min_value=30, max_value=300, value=100, step=10,
            key="hetero_n_slider"
        )

    # Generate heteroscedastic data
    np.random.seed(42)
    x = np.random.uniform(0, 10, n_samples)

    # Create errors with variance that increases with x
    error_sd = 0.5 + hetero_strength * x
    epsilon = np.random.normal(0, 1, n_samples) * error_sd

    y = 2 + 3 * x + epsilon

    # Fit OLS model
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # Calculate correct standard errors (HC1)
    model_robust = sm.OLS(y, X).fit(cov_type='HC1')

    col1, col2 = st.columns([1, 1])

    with col1:
        # Scatter plot with regression line
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, alpha=0.6)
        ax.plot(np.sort(x), model.predict()[np.argsort(x)], color='red', label='OLS regression line')

        # Add confidence intervals (using both regular and robust SE)
        x_sorted = np.sort(x)
        X_sorted = sm.add_constant(x_sorted)

        # Regular CIs
        y_pred = model.predict(X_sorted)
        y_ci = model.get_prediction(X_sorted).conf_int(alpha=0.05)
        ax.fill_between(x_sorted, y_ci[:, 0], y_ci[:, 1], color='red', alpha=0.1, label='95% CI (Regular)')

        # Robust CIs
        y_ci_robust = model_robust.get_prediction(X_sorted).conf_int(alpha=0.05)
        ax.fill_between(x_sorted, y_ci_robust[:, 0], y_ci_robust[:, 1], color='blue', alpha=0.1,
                        label='95% CI (Robust)')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Regression with Heteroscedastic Errors")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    with col2:
        # Plot residuals
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, model.resid, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='-')

        # Add smooth trend line for residuals
        from scipy.stats import binned_statistic

        bin_means, bin_edges, _ = binned_statistic(x, np.abs(model.resid), statistic='mean', bins=10)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, bin_means, color='red', linewidth=2, label='Mean absolute residual')

        ax.set_xlabel("X")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs. X (Heteroscedasticity)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Display regression results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Regression Results Comparison")
        st.markdown(f"""
        **Regular OLS Standard Errors**:
        - β₁ estimate: {model.params[1]:.4f}
        - Standard error: {model.bse[1]:.4f}
        - t-statistic: {model.tvalues[1]:.4f}
        - p-value: {model.pvalues[1]:.4f}
        - 95% CI: [{model.conf_int().iloc[1, 0]:.4f}, {model.conf_int().iloc[1, 1]:.4f}]

        **Robust (HC1) Standard Errors**:
        - β₁ estimate: {model_robust.params[1]:.4f} (same)
        - Robust standard error: {model_robust.bse[1]:.4f}
        - Robust t-statistic: {model_robust.tvalues[1]:.4f}
        - Robust p-value: {model_robust.pvalues[1]:.4f}
        - Robust 95% CI: [{model_robust.conf_int().iloc[1, 0]:.4f}, {model_robust.conf_int().iloc[1, 1]:.4f}]

        **Ratio of Robust to Regular SE**: {model_robust.bse[1] / model.bse[1]:.3f}
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # Monte Carlo simulation
    st.markdown('<div class="section-header">Monte Carlo Simulation: Impact on Inference</div>', unsafe_allow_html=True)

    if st.button("Run Monte Carlo Simulation for Heteroscedasticity", key="hetero_mc_button"):
        with st.spinner("Running simulation..."):
            # Function to generate heteroscedastic data
            def generate_heteroscedastic_data(n=100, beta0=2, beta1=3, hetero_strength=0.5, seed=42):
                np.random.seed(seed)
                x = np.random.uniform(0, 10, n)
                error_sd = 0.5 + hetero_strength * x
                epsilon = np.random.normal(0, 1, n) * error_sd
                y = beta0 + beta1 * x + epsilon
                return x, y, epsilon


            # Function to generate homoscedastic data
            def generate_homoscedastic_data(n=100, beta0=2, beta1=3, error_sd=1, seed=42):
                np.random.seed(seed)
                x = np.random.uniform(0, 10, n)
                epsilon = np.random.normal(0, error_sd, n)
                y = beta0 + beta1 * x + epsilon
                return x, y, epsilon


            # Run simulations
            n_simulations = 1000

            # Arrays to store results
            betas_homo = []
            ses_homo = []
            robust_ses_homo = []

            betas_hetero = []
            ses_hetero = []
            robust_ses_hetero = []

            for i in range(n_simulations):
                # Homoscedastic case
                x, y, _ = generate_homoscedastic_data(n=n_samples, seed=i)
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                model_robust = sm.OLS(y, X).fit(cov_type='HC1')

                betas_homo.append(model.params[1])
                ses_homo.append(model.bse[1])
                robust_ses_homo.append(model_robust.bse[1])

                # Heteroscedastic case
                x, y, _ = generate_heteroscedastic_data(n=n_samples, hetero_strength=hetero_strength, seed=i)
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                model_robust = sm.OLS(y, X).fit(cov_type='HC1')

                betas_hetero.append(model.params[1])
                ses_hetero.append(model.bse[1])
                robust_ses_hetero.append(model_robust.bse[1])

            # Convert to numpy arrays
            betas_homo = np.array(betas_homo)
            ses_homo = np.array(ses_homo)
            robust_ses_homo = np.array(robust_ses_homo)

            betas_hetero = np.array(betas_hetero)
            ses_hetero = np.array(ses_hetero)
            robust_ses_hetero = np.array(robust_ses_hetero)

            # Calculate t-stats and CI coverage
            t_stats_homo = (betas_homo - 3) / ses_homo
            t_stats_hetero = (betas_hetero - 3) / ses_hetero

            robust_t_stats_homo = (betas_homo - 3) / robust_ses_homo
            robust_t_stats_hetero = (betas_hetero - 3) / robust_ses_hetero

            # Critical value for 95% CI
            crit_val = stats.t.ppf(0.975, n_samples - 2)

            # Calculate CI coverage
            ci_coverage_homo = np.mean((betas_homo - crit_val * ses_homo <= 3) &
                                       (betas_homo + crit_val * ses_homo >= 3))
            ci_coverage_hetero = np.mean((betas_hetero - crit_val * ses_hetero <= 3) &
                                         (betas_hetero + crit_val * ses_hetero >= 3))

            robust_ci_coverage_homo = np.mean((betas_homo - crit_val * robust_ses_homo <= 3) &
                                              (betas_homo + crit_val * robust_ses_homo >= 3))
            robust_ci_coverage_hetero = np.mean((betas_hetero - crit_val * robust_ses_hetero <= 3) &
                                                (betas_hetero + crit_val * robust_ses_hetero >= 3))

            col1, col2 = st.columns([1, 1])

            with col1:
                # Compare distribution of coefficient estimates
                fig = compare_distributions(betas_homo, betas_hetero,
                                            "Sampling Distribution of β₁")
                st.pyplot(fig)
                st.markdown(
                    '<div class="caption">Distribution of β₁ under homoscedasticity vs. heteroscedasticity.</div>',
                    unsafe_allow_html=True)

                # Compare distribution of standard errors
                fig = compare_distributions(ses_homo, ses_hetero,
                                            "Distribution of Standard Errors")
                st.pyplot(fig)
                st.markdown(
                    '<div class="caption">Distribution of standard errors under homoscedasticity vs. heteroscedasticity.</div>',
                    unsafe_allow_html=True)

            with col2:
                # Compare distribution of t-statistics
                fig = compare_distributions(t_stats_homo, t_stats_hetero,
                                            "Distribution of t-statistics")
                st.pyplot(fig)
                st.markdown(
                    '<div class="caption">Distribution of t-statistics under homoscedasticity vs. heteroscedasticity.</div>',
                    unsafe_allow_html=True)

                # Compare distribution of robust standard errors
                fig = compare_distributions(robust_ses_homo, robust_ses_hetero,
                                            "Distribution of Robust Standard Errors")
                st.pyplot(fig)
                st.markdown('<div class="caption">Distribution of robust standard errors under both conditions.</div>',
                            unsafe_allow_html=True)

            # Display key findings
            st.markdown('<div class="important-note">', unsafe_allow_html=True)
            st.markdown(f"""
            **Key Findings**:

            1. **Impact on coefficient estimates**:
               - Average β₁ (homoscedastic): {np.mean(betas_homo):.4f}
               - Average β₁ (heteroscedastic): {np.mean(betas_hetero):.4f}
               - Standard deviation of β₁ (homoscedastic): {np.std(betas_homo):.4f}
               - Standard deviation of β₁ (heteroscedastic): {np.std(betas_hetero):.4f}

            2. **Impact on standard errors**:
               - Average SE (homoscedastic): {np.mean(ses_homo):.4f}
               - Average SE (heteroscedastic): {np.mean(ses_hetero):.4f}
               - **True to estimated SE ratio (heteroscedastic): {np.std(betas_hetero) / np.mean(ses_hetero):.3f}**

            3. **Confidence interval coverage** (should be 95%):
               - Conventional CI coverage (homoscedastic): {ci_coverage_homo * 100:.1f}%
               - Conventional CI coverage (heteroscedastic): {ci_coverage_hetero * 100:.1f}%
               - Robust CI coverage (homoscedastic): {robust_ci_coverage_homo * 100:.1f}%
               - Robust CI coverage (heteroscedastic): {robust_ci_coverage_hetero * 100:.1f}%

            4. **Type I error rates** (for H₀: β₁ = 3, α = 0.05):
               - Conventional tests (homoscedastic): {np.mean(np.abs(t_stats_homo) > crit_val) * 100:.1f}%
               - Conventional tests (heteroscedastic): {np.mean(np.abs(t_stats_hetero) > crit_val) * 100:.1f}%
               - Robust tests (homoscedastic): {np.mean(np.abs(robust_t_stats_homo) > crit_val) * 100:.1f}%
               - Robust tests (heteroscedastic): {np.mean(np.abs(robust_t_stats_hetero) > crit_val) * 100:.1f}%
            """)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Detection and Solutions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### How to Detect Heteroscedasticity")
        st.markdown("""
        1. **Visual inspection**:
           - Residuals vs. fitted values plot
           - Residuals vs. independent variables plots
           - Scale-location plots

        2. **Statistical tests**:
           - Breusch-Pagan test
           - White test
           - Goldfeld-Quandt test
           - Park test
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Solutions to Heteroscedasticity")
        st.markdown("""
        1. **Robust standard errors**:
           - HC0, HC1, HC2, HC3 variants
           - Huber-White sandwich estimators
           - Clustered standard errors

        2. **Transformation of variables**:
           - Log transformation
           - Box-Cox transformation

        3. **Weighted Least Squares (WLS)**:
           - Weight observations by inverse of error variance

        4. **Generalized Least Squares (GLS)**:
           - Accounts for non-constant error variance
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Autocorrelation Violation
elif page == "Autocorrelation Violation":
    st.markdown('<div class="sub-header">Violation of Independence: Autocorrelation</div>', unsafe_allow_html=True)

    st.markdown("""
    The independence assumption requires that error terms are uncorrelated across observations. Autocorrelation (or serial correlation) 
    occurs when error terms are correlated across observations, typically in time series or spatial data.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Effects on OLS Estimators</div>', unsafe_allow_html=True)
        st.markdown("""
        When errors are autocorrelated:

        1. **Unbiasedness Preserved**: OLS estimators remain unbiased (if other assumptions hold)
        2. **Efficiency Lost**: OLS is no longer BLUE (Best Linear Unbiased Estimator)
        3. **Incorrect Standard Errors**: Usually underestimated with positive autocorrelation
        4. **Spurious Regression**: Risk of finding relationships that don't exist
        """)

        st.markdown('<div class="section-header">Impact on Statistical Inference</div>', unsafe_allow_html=True)
        st.markdown("""
        - **t-statistics**: Typically inflated with positive autocorrelation
        - **Confidence Intervals**: Too narrow, giving false precision
        - **Hypothesis Tests**: Inflated Type I error rates
        - **R-squared**: Artificially high, giving illusion of good fit
        - **Durbin-Watson Statistic**: No longer close to 2 (expected value under independence)
        """)

    with col2:
        st.markdown('<div class="important-note">', unsafe_allow_html=True)
        st.markdown("""
        **Why Students Should Care**:

        Autocorrelation leads to:

        - Overconfidence in results
        - Spurious significance
        - Inefficient forecasts
        - Inappropriate policy recommendations

        Especially common in:
        - Time series data
        - Spatial data
        - Hierarchical/clustered data
        - Panel/longitudinal data
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Mathematical Expression")
        st.markdown(r'''
        For positive autocorrelation (AR(1) process):

        $$\varepsilon_t = \rho \varepsilon_{t-1} + u_t$$

        Where:
        - $\rho$ is the autocorrelation coefficient (-1 < $\rho$ < 1)
        - $u_t$ is white noise

        The covariance matrix of errors becomes:

        $$\Omega = \sigma^2
        \begin{pmatrix} 
        1 & \rho & \rho^2 & \ldots & \rho^{n-1} \\
        \rho & 1 & \rho & \ldots & \rho^{n-2} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        \rho^{n-1} & \rho^{n-2} & \rho^{n-3} & \ldots & 1
        \end{pmatrix}$$
        ''')
        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive demonstration
    st.markdown('<div class="section-header">Interactive Demonstration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        autocorr_coef = st.slider(
            "Autocorrelation Coefficient (ρ)",
            min_value=-0.9, max_value=0.9, value=0.7, step=0.1,
            key="autocorr_slider"
        )

    with col2:
        n_samples = st.slider(
            "Sample Size",
            min_value=30, max_value=200, value=100, step=10,
            key="autocorr_n_slider"
        )

    # Generate autocorrelated data
    x, y, epsilon = generate_autocorrelated_data(n=n_samples, rho=autocorr_coef)

    # Fit OLS model
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # Calculate Newey-West standard errors
    model_nw = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

    # Calculate Durbin-Watson statistic
    dw_stat = sm.stats.stattools.durbin_watson(model.resid)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Plot regression
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, alpha=0.6)
        ax.plot(np.sort(x), model.predict()[np.argsort(x)], color='red', label='OLS regression line')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Regression with Autocorrelated Errors (ρ = {autocorr_coef})")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Plot error terms
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epsilon, marker='o', linestyle='-', markersize=4, alpha=0.7)
        ax.set_xlabel("Observation Number")
        ax.set_ylabel("Error Term")
        ax.set_title(f"Error Terms Over Time (ρ = {autocorr_coef})")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    with col2:
        # Plot autocorrelation function
        fig, ax = plt.subplots(figsize=(10, 6))
        sm.graphics.tsa.plot_acf(epsilon, lags=20, alpha=0.05, ax=ax)
        ax.set_title("Autocorrelation Function of Error Terms")
        st.pyplot(fig)

        # Display regression results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Regression Results Comparison")
        st.markdown(f"""
        **Regular OLS Standard Errors**:
        - β₁ estimate: {model.params[1]:.4f}
        - Standard error: {model.bse[1]:.4f}
        - t-statistic: {model.tvalues[1]:.4f}
        - p-value: {model.pvalues[1]:.4f}

        **Newey-West HAC Standard Errors**:
        - β₁ estimate: {model_nw.params[1]:.4f} (same)
        - Robust standard error: {model_nw.bse[1]:.4f}
        - Robust t-statistic: {model_nw.tvalues[1]:.4f}
        - Robust p-value: {model_nw.pvalues[1]:.4f}

        **Ratio of HAC to Regular SE**: {model_nw.bse[1] / model.bse[1]:.3f}

        **Durbin-Watson Statistic**: {dw_stat:.4f}
        - DW ≈ 2: No autocorrelation
        - DW < 2: Positive autocorrelation
        - DW > 2: Negative autocorrelation
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # Monte Carlo simulation
    st.markdown('<div class="section-header">Monte Carlo Simulation: Impact on Inference</div>', unsafe_allow_html=True)

    if st.button("Run Monte Carlo Simulation for Autocorrelation", key="autocorr_mc_button"):
        with st.spinner("Running simulation..."):
            # Function to generate independent data (no autocorrelation)
            def generate_independent_data(n=100, beta0=2, beta1=3, error_sd=1, seed=42):
                np.random.seed(seed)
                x = np.random.uniform(0, 10, n)
                epsilon = np.random.normal(0, error_sd, n)
                y = beta0 + beta1 * x + epsilon
                return x, y, epsilon


            # Run simulations
            n_simulations = 1000

            # Arrays to store results
            betas_indep = []
            ses_indep = []
            nw_ses_indep = []

            betas_autocorr = []
            ses_autocorr = []
            nw_ses_autocorr = []

            dw_stats_indep = []
            dw_stats_autocorr = []

            for i in range(n_simulations):
                # Independent case
                x, y, _ = generate_independent_data(n=n_samples, seed=i)
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                model_nw = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

                betas_indep.append(model.params[1])
                ses_indep.append(model.bse[1])
                nw_ses_indep.append(model_nw.bse[1])
                dw_stats_indep.append(sm.stats.stattools.durbin_watson(model.resid))

                # Autocorrelated case
                x, y, _ = generate_autocorrelated_data(n=n_samples, rho=autocorr_coef, seed=i)
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                model_nw = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

                betas_autocorr.append(model.params[1])
                ses_autocorr.append(model.bse[1])
                nw_ses_autocorr.append(model_nw.bse[1])
                dw_stats_autocorr.append(sm.stats.stattools.durbin_watson(model.resid))

            # Convert to numpy arrays
            betas_indep = np.array(betas_indep)
            ses_indep = np.array(ses_indep)
            nw_ses_indep = np.array(nw_ses_indep)
            dw_stats_indep = np.array(dw_stats_indep)

            betas_autocorr = np.array(betas_autocorr)
            ses_autocorr = np.array(ses_autocorr)
            nw_ses_autocorr = np.array(nw_ses_autocorr)
            dw_stats_autocorr = np.array(dw_stats_autocorr)

            # Calculate t-stats and CI coverage
            t_stats_indep = (betas_indep - 3) / ses_indep
            t_stats_autocorr = (betas_autocorr - 3) / ses_autocorr

            nw_t_stats_indep = (betas_indep - 3) / nw_ses_indep
            nw_t_stats_autocorr = (betas_autocorr - 3) / nw_ses_autocorr

            # Critical value for 95% CI
            crit_val = stats.t.ppf(0.975, n_samples - 2)

            # Calculate CI coverage
            ci_coverage_indep = np.mean((betas_indep - crit_val * ses_indep <= 3) &
                                        (betas_indep + crit_val * ses_indep >= 3))
            ci_coverage_autocorr = np.mean((betas_autocorr - crit_val * ses_autocorr <= 3) &
                                           (betas_autocorr + crit_val * ses_autocorr >= 3))

            nw_ci_coverage_indep = np.mean((betas_indep - crit_val * nw_ses_indep <= 3) &
                                           (betas_indep + crit_val * nw_ses_indep >= 3))
            nw_ci_coverage_autocorr = np.mean((betas_autocorr - crit_val * nw_ses_autocorr <= 3) &
                                              (betas_autocorr + crit_val * nw_ses_autocorr >= 3))

            col1, col2 = st.columns([1, 1])

            with col1:
                # Compare distribution of coefficient estimates
                fig = compare_distributions(betas_indep, betas_autocorr,
                                            "Sampling Distribution of β₁")
                st.pyplot(fig)
                st.markdown('<div class="caption">Distribution of β₁ under independence vs. autocorrelation.</div>',
                            unsafe_allow_html=True)

                # Compare distribution of standard errors
                fig = compare_distributions(ses_indep, ses_autocorr,
                                            "Distribution of Standard Errors")
                st.pyplot(fig)
                st.markdown(
                    '<div class="caption">Distribution of standard errors under independence vs. autocorrelation.</div>',
                    unsafe_allow_html=True)

            with col2:
                # Compare distribution of t-statistics
                fig = compare_distributions(t_stats_indep, t_stats_autocorr,
                                            "Distribution of t-statistics")
                st.pyplot(fig)
                st.markdown(
                    '<div class="caption">Distribution of t-statistics under independence vs. autocorrelation.</div>',
                    unsafe_allow_html=True)

                # Compare distribution of Durbin-Watson statistics
                fig = compare_distributions(dw_stats_indep, dw_stats_autocorr,
                                            "Distribution of Durbin-Watson Statistics")
                st.pyplot(fig)
                st.markdown(
                    '<div class="caption">Distribution of Durbin-Watson statistics under independence vs. autocorrelation.</div>',
                    unsafe_allow_html=True)

            # Display key findings
            st.markdown('<div class="important-note">', unsafe_allow_html=True)
            st.markdown(f"""
            **Key Findings**:

            1. **Impact on coefficient estimates**:
               - Average β₁ (independent): {np.mean(betas_indep):.4f}
               - Average β₁ (autocorrelated, ρ={autocorr_coef}): {np.mean(betas_autocorr):.4f}
               - Standard deviation of β₁ (independent): {np.std(betas_indep):.4f}
               - Standard deviation of β₁ (autocorrelated): {np.std(betas_autocorr):.4f}

            2. **Impact on standard errors**:
               - Average SE (independent): {np.mean(ses_indep):.4f}
               - Average SE (autocorrelated): {np.mean(ses_autocorr):.4f}
               - **True to estimated SE ratio (autocorrelated): {np.std(betas_autocorr) / np.mean(ses_autocorr):.3f}**

            3. **Impact of HAC standard errors**:
               - Average HAC SE (independent): {np.mean(nw_ses_indep):.4f}
               - Average HAC SE (autocorrelated): {np.mean(nw_ses_autocorr):.4f}
               - **True to estimated HAC SE ratio (autocorrelated): {np.std(betas_autocorr) / np.mean(nw_ses_autocorr):.3f}**

            4. **Confidence interval coverage** (should be 95%):
               - Conventional CI coverage (independent): {ci_coverage_indep * 100:.1f}%
               - Conventional CI coverage (autocorrelated): {ci_coverage_autocorr * 100:.1f}%
               - HAC CI coverage (independent): {nw_ci_coverage_indep * 100:.1f}%
               - HAC CI coverage (autocorrelated): {nw_ci_coverage_autocorr * 100:.1f}%

            5. **Type I error rates** (for H₀: β₁ = 3, α = 0.05):
               - Conventional tests (independent): {np.mean(np.abs(t_stats_indep) > crit_val) * 100:.1f}%
               - Conventional tests (autocorrelated): {np.mean(np.abs(t_stats_autocorr) > crit_val) * 100:.1f}%
               - HAC tests (independent): {np.mean(np.abs(nw_t_stats_indep) > crit_val) * 100:.1f}%
               - HAC tests (autocorrelated): {np.mean(np.abs(nw_t_stats_autocorr) > crit_val) * 100:.1f}%

            6. **Durbin-Watson statistics**:
               - Average DW (independent): {np.mean(dw_stats_indep):.4f}
               - Average DW (autocorrelated): {np.mean(dw_stats_autocorr):.4f}
            """)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Detection and Solutions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### How to Detect Autocorrelation")
        st.markdown("""
        1. **Visual inspection**:
           - Plot residuals over time/sequence
           - Autocorrelation function (ACF) plot
           - Partial autocorrelation function (PACF) plot

        2. **Statistical tests**:
           - Durbin-Watson test
           - Breusch-Godfrey test
           - Ljung-Box Q test
           - Runs test
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Solutions to Autocorrelation")
        st.markdown("""
        1. **Robust standard errors**:
           - Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) estimators
           - Andrews automatic bandwidth selection

        2. **Model transformation**:
           - First-differencing
           - Cochrane-Orcutt procedure
           - Prais-Winsten transformation

        3. **Dynamic modeling**:
           - Include lagged dependent variables
           - ARIMA modeling
           - Error Correction Models (ECM)

        4. **Generalized Least Squares (GLS)**:
           - Feasible GLS accounting for autocorrelation structure
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Multicollinearity
elif page == "Multicollinearity":
    st.markdown('<div class="sub-header">Violation of No Multicollinearity Assumption</div>', unsafe_allow_html=True)

    st.markdown("""
    Multicollinearity occurs when two or more independent variables in a regression model are highly correlated. Perfect multicollinearity 
    makes the model unidentifiable, while high multicollinearity causes estimation problems.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Effects on OLS Estimators</div>', unsafe_allow_html=True)
        st.markdown("""
        When independent variables are highly correlated:

        1. **Unbiasedness Preserved**: OLS estimators remain unbiased
        2. **Increased Variance**: Standard errors become inflated
        3. **Unstable Coefficients**: Small changes in data can cause large changes in estimates
        4. **Difficulty in Attribution**: Hard to separate effects of correlated variables
        """)

        st.markdown('<div class="section-header">Impact on Statistical Inference</div>', unsafe_allow_html=True)
        st.markdown("""
        - **t-statistics**: Reduced due to inflated standard errors
        - **Confidence Intervals**: Wider, less precise
        - **Hypothesis Tests**: Reduced power to detect true effects (higher Type II error rates)
        - **F-test vs. t-tests**: F-test for joint significance may be significant while individual t-tests are not
        - **Coefficient Signs**: May be counterintuitive or change unexpectedly
        """)

    with col2:
        st.markdown('<div class="important-note">', unsafe_allow_html=True)
        st.markdown("""
        **Why Students Should Care**:

        Multicollinearity leads to:

        - Difficulty identifying important variables
        - Reduced statistical power
        - Misleading coefficients
        - Unstable predictive models
        - False negatives in hypothesis testing

        Common in:
        - Economic data
        - Survey data with similar questions
        - Models with polynomial terms
        - Time series with trends
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Mathematical Expression")
        st.markdown(r'''
        The variance of coefficients under multicollinearity:

        $$Var(\hat{\beta}_j) = \frac{\sigma^2}{(1-R_j^2)\sum(X_j - \bar{X}_j)^2}$$

        Where $R_j^2$ is the R-squared from regressing $X_j$ on all other predictors.

        As $R_j^2$ approaches 1 (high multicollinearity), the variance approaches infinity.

        The Variance Inflation Factor (VIF) for variable j is:

        $$VIF_j = \frac{1}{1-R_j^2}$$
        ''')
        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive demonstration
    st.markdown('<div class="section-header">Interactive Demonstration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        correlation = st.slider(
            "Correlation between X₁ and X₂",
            min_value=0.0, max_value=0.99, value=0.8, step=0.05,
            key="multicollinearity_slider"
        )

    with col2:
        n_samples = st.slider(
            "Sample Size",
            min_value=30, max_value=300, value=100, step=10,
            key="multicollinearity_n_slider"
        )

    # Generate multicollinear data
    x1, x2, y, epsilon = generate_multicollinear_data(n=n_samples, correlation=correlation)

    # Fit OLS model
    X = sm.add_constant(np.column_stack((x1, x2)))
    model = sm.OLS(y, X).fit()

    # Calculate VIFs
    vif_data = pd.DataFrame()
    vif_data["Variable"] = ["X₁", "X₂"]
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(1, 3)]

    col1, col2 = st.columns([1, 1])

    with col1:
        # 3D scatter plot
        fig = px.scatter_3d(
            x=x1, y=x2, z=y,
            labels={'x': 'X₁', 'y': 'X₂', 'z': 'Y'},
            title=f"3D Visualization with Correlation = {correlation:.2f}"
        )

        # Add the fitted plane
        # Create a meshgrid
        x1_range = np.linspace(min(x1), max(x1), 20)
        x2_range = np.linspace(min(x2), max(x2), 20)
        x1_mg, x2_mg = np.meshgrid(x1_range, x2_range)

        # Calculate the predicted values
        X_mesh = sm.add_constant(np.column_stack((x1_mg.flatten(), x2_mg.flatten())))
        y_pred = model.predict(X_mesh).reshape(x1_mg.shape)

        # Add the surface
        fig.add_trace(
            go.Surface(x=x1_range, y=x2_range, z=y_pred, opacity=0.7,
                       colorscale='Viridis', showscale=False)
        )

        fig.update_layout(
            scene=dict(
                xaxis_title='X₁',
                yaxis_title='X₂',
                zaxis_title='Y'
            ),
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Scatter plot of X1 vs X2
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x1, x2, alpha=0.6)
        ax.set_xlabel("X₁")
        ax.set_ylabel("X₂")
        ax.set_title(f"Relationship between X₁ and X₂ (r = {correlation:.2f})")

        # Add regression line
        x1_sorted = np.sort(x1)
        ax.plot(x1_sorted, np.poly1d(np.polyfit(x1, x2, 1))(x1_sorted), color='red')

        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Display VIFs
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Variance Inflation Factors (VIFs)")
        st.markdown("""
        VIF > 10 indicates problematic multicollinearity
        VIF > 5 indicates moderate multicollinearity
        """)
        st.table(vif_data.set_index("Variable"))
        st.markdown("</div>", unsafe_allow_html=True)

        # Display regression results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Regression Results")
        st.markdown(f"""
        **Coefficients and Standard Errors**:
        - β₁: {model.params[1]:.4f} (SE: {model.bse[1]:.4f})
        - β₂: {model.params[2]:.4f} (SE: {model.bse[2]:.4f})

        **t-statistics and p-values**:
        - β₁: t = {model.tvalues[1]:.4f}, p = {model.pvalues[1]:.4f}
        - β₂: t = {model.tvalues[2]:.4f}, p = {model.pvalues[2]:.4f}

        **F-statistic**: {model.fvalue:.4f} (p-value: {model.f_pvalue:.4f})

        **R-squared**: {model.rsquared:.4f}
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # Monte Carlo simulation
    st.markdown('<div class="section-header">Monte Carlo Simulation: Impact of Multicollinearity</div>',
                unsafe_allow_html=True)

    if st.button("Run Monte Carlo Simulation for Multicollinearity", key="multicollinearity_mc_button"):
        with st.spinner("Running simulation..."):
            # Run simulations with different correlation levels
            correlations = [0.0, 0.5, 0.8, 0.95]
            results = {}

            for corr in correlations:
                betas, t_stats, std_errors = run_monte_carlo(
                    generate_multicollinear_data,
                    n_simulations=1000,
                    n_samples=n_samples,
                    correlation=corr
                )

                results[corr] = {
                    "betas_1": betas[:, 0],  # Beta1
                    "betas_2": betas[:, 1],  # Beta2
                    "t_stats_1": t_stats[:, 0],  # t-stat for Beta1
                    "t_stats_2": t_stats[:, 1],  # t-stat for Beta2
                    "std_errors_1": std_errors[:, 0],  # SE for Beta1
                    "std_errors_2": std_errors[:, 1]  # SE for Beta2
                }

            col1, col2 = st.columns([1, 1])

            with col1:
                # Plot distribution of Beta1 for different correlation levels
                fig, ax = plt.subplots(figsize=(10, 6))
                for corr in correlations:
                    sns.kdeplot(results[corr]["betas_1"], ax=ax, label=f"r = {corr}")

                ax.axvline(3, color='black', linestyle='--', alpha=0.7, label="True β₁ = 3")
                ax.set_xlabel("Estimated β₁")
                ax.set_title("Sampling Distribution of β₁ for Different Correlation Levels")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)

                # Plot distribution of standard errors for Beta1
                fig, ax = plt.subplots(figsize=(10, 6))
                for corr in correlations:
                    sns.kdeplot(results[corr]["std_errors_1"], ax=ax, label=f"r = {corr}")

                ax.set_xlabel("Standard Error of β₁")
                ax.set_title("Distribution of Standard Errors for β₁")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)

            with col2:
                # Plot distribution of t-statistics for Beta1
                fig, ax = plt.subplots(figsize=(10, 6))
                for corr in correlations:
                    sns.kdeplot(results[corr]["t_stats_1"], ax=ax, label=f"r = {corr}")

                # Add critical values
                crit_val = stats.t.ppf(0.975, n_samples - 3)
                ax.axvline(-crit_val, color='red', linestyle='--', alpha=0.7)
                ax.axvline(crit_val, color='red', linestyle='--', alpha=0.7)

                ax.set_xlabel("t-statistic for β₁")
                ax.set_title("Distribution of t-statistics for β₁")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)

                # Create a scatterplot of Beta1 vs Beta2 estimates
                fig, ax = plt.subplots(figsize=(10, 6))
                for corr in correlations:
                    ax.scatter(results[corr]["betas_1"], results[corr]["betas_2"],
                               alpha=0.3, label=f"r = {corr}")

                ax.axvline(3, color='black', linestyle='--', alpha=0.5, label="True β₁ = 3")
                ax.axhline(1.5, color='black', linestyle='--', alpha=0.5, label="True β₂ = 1.5")
                ax.set_xlabel("Estimated β₁")
                ax.set_ylabel("Estimated β₂")
                ax.set_title("Relationship Between β₁ and β₂ Estimates")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)

            # Calculate summary statistics
            summary_data = []
            for corr in correlations:
                # Mean estimates
                mean_beta1 = np.mean(results[corr]["betas_1"])
                mean_beta2 = np.mean(results[corr]["betas_2"])

                # Standard deviations of estimates
                sd_beta1 = np.std(results[corr]["betas_1"])
                sd_beta2 = np.std(results[corr]["betas_2"])

                # Mean standard errors
                mean_se_beta1 = np.mean(results[corr]["std_errors_1"])
                mean_se_beta2 = np.mean(results[corr]["std_errors_2"])

                # Correlation between beta1 and beta2 estimates
                corr_betas = np.corrcoef(results[corr]["betas_1"], results[corr]["betas_2"])[0, 1]

                # Calculate rejection rates
                crit_val = stats.t.ppf(0.975, n_samples - 3)
                reject_beta1 = np.mean(np.abs(results[corr]["t_stats_1"]) > crit_val)
                reject_beta2 = np.mean(np.abs(results[corr]["t_stats_2"]) > crit_val)

                summary_data.append({
                    "Correlation": corr,
                    "Mean β₁": mean_beta1,
                    "SD of β₁": sd_beta1,
                    "Mean SE of β₁": mean_se_beta1,
                    "Mean β₂": mean_beta2,
                    "SD of β₂": sd_beta2,
                    "Mean SE of β₂": mean_se_beta2,
                    "Corr(β₁, β₂)": corr_betas,
                    "Rejection Rate β₁": reject_beta1,
                    "Rejection Rate β₂": reject_beta2
                })

            # Create a summary table
            summary_df = pd.DataFrame(summary_data).set_index("Correlation")
            summary_df = summary_df.round(4)

            st.markdown('<div class="important-note">', unsafe_allow_html=True)
            st.markdown("### Key Findings from the Simulation")
            st.markdown("""
            As correlation between predictors increases:

            1. **Estimates remain unbiased** (mean estimates close to true values)
            2. **Standard errors increase dramatically**
            3. **Sampling variability increases** (wider spread of coefficient estimates)
            4. **Strong negative correlation** between coefficient estimates
            5. **Statistical power decreases** (lower rejection rates for true effects)
            """)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### Summary Statistics from Simulation")
            st.table(summary_df.T)

    st.markdown('<div class="section-header">Detection and Solutions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### How to Detect Multicollinearity")
        st.markdown("""
        1. **Correlation matrix**: Check for high correlations between predictors

        2. **Variance Inflation Factor (VIF)**:
           - VIF > 10: Severe multicollinearity
           - VIF > 5: Moderate multicollinearity

        3. **Condition number** of X'X matrix:
           - > 30: Moderate multicollinearity
           - > 100: Severe multicollinearity

        4. **Warning signs in results**:
           - High R-squared but few significant t-statistics
           - Coefficients with unexpected signs
           - Large changes in coefficients when variables are added/removed
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Solutions to Multicollinearity")
        st.markdown("""
        1. **Remove some correlated predictors**:
           - Subset selection
           - Stepwise selection

        2. **Combine predictors**:
           - Principal Component Analysis (PCA)
           - Factor analysis
           - Create composite indices

        3. **Regularization techniques**:
           - Ridge regression
           - LASSO regression
           - Elastic Net

        4. **Increase sample size**:
           - Reduces standard errors
           - May not be feasible

        5. **Center variables** (for polynomial terms, interactions)
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Endogeneity
elif page == "Endogeneity":
    st.markdown('<div class="sub-header">Violation of No Endogeneity Assumption</div>', unsafe_allow_html=True)

    st.markdown("""
    Endogeneity occurs when an independent variable is correlated with the error term in the regression model. 
    This is perhaps the most serious violation of OLS assumptions, as it leads to biased and inconsistent estimates.
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Effects on OLS Estimators</div>', unsafe_allow_html=True)
        st.markdown("""
        When independent variables are correlated with error terms:

        1. **Biased Estimators**: OLS estimates are systematically biased
        2. **Inconsistency**: Bias doesn't disappear even with large samples
        3. **Incorrect Causal Inference**: Estimated relationships don't represent causal effects
        4. **All Coefficients Affected**: Bias can spread to coefficients of exogenous variables
        """)

        st.markdown('<div class="section-header">Impact on Statistical Inference</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Misleading t-statistics and p-values**: Based on biased estimates
        - **Invalid Confidence Intervals**: Don't contain the true parameter values
        - **Unpredictable Direction of Bias**: Can be positive or negative
        - **Incorrect Policy Recommendations**: Based on faulty causal interpretations
        """)

    with col2:
        st.markdown('<div class="important-note">', unsafe_allow_html=True)
        st.markdown("""
        **Why Students Should Care**:

        Endogeneity leads to:

        - Wrong conclusions about causal relationships
        - Invalid policy recommendations
        - Misleading scientific findings
        - Unreliable predictions

        Common sources:
        - Omitted variables
        - Measurement error
        - Simultaneity/reverse causality
        - Sample selection bias
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Mathematical Expression")
        st.markdown(r'''
        Suppose the true model is:

        $$Y = \beta_0 + \beta_1 X + \varepsilon$$

        Under endogeneity, $Cov(X, \varepsilon) \neq 0$

        The OLS estimator has expectation:

        $$E[\hat{\beta}_1] = \beta_1 + \frac{Cov(X, \varepsilon)}{Var(X)}$$

        The second term represents the endogeneity bias.
        ''')
        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive demonstration
    st.markdown('<div class="section-header">Interactive Demonstration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        endogeneity_level = st.slider(
            "Degree of Endogeneity",
            min_value=0.0, max_value=0.9, value=0.5, step=0.1,
            key="endogeneity_slider"
        )

    with col2:
        n_samples = st.slider(
            "Sample Size",
            min_value=30, max_value=300, value=100, step=10,
            key="endogeneity_n_slider"
        )

    # Generate endogenous data
    x, y, epsilon, unobserved = generate_endogenous_data(n=n_samples, endogeneity=endogeneity_level)

    # Fit OLS model (ignoring endogeneity)
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # Fit "correct" model (if we could observe the unobserved variable)
    X_correct = sm.add_constant(np.column_stack((x, unobserved)))
    model_correct = sm.OLS(y, X_correct).fit()

    col1, col2 = st.columns([1, 1])

    with col1:
        # Scatter plot with regression line
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(x, y, c=unobserved, cmap='viridis', alpha=0.7)

        # Sort for smooth line plotting
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]

        # Plot OLS fit (ignoring endogeneity)
        ax.plot(x_sorted, model.predict()[sort_idx], color='red',
                label=f'OLS fit (β̂₁ = {model.params[1]:.3f})')

        # Calculate true relationship (β₁ = 3)
        true_y = 2 + 3 * x_sorted
        ax.plot(x_sorted, true_y, color='black', linestyle='--',
                label=f'True relationship (β₁ = 3.000)')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Regression with Endogeneity (correlation = {endogeneity_level:.2f})")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add colorbar for unobserved variable
        cbar = plt.colorbar(scatter)
        cbar.set_label('Unobserved Variable')

        st.pyplot(fig)

    with col2:
        # Plot relationship between x and the error term
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, epsilon, alpha=0.6)

        # Add trend line
        z = np.polyfit(x, epsilon, 1)
        p = np.poly1d(z)
        ax.plot(np.sort(x), p(np.sort(x)), "r--", alpha=0.8)

        ax.set_xlabel("X")
        ax.set_ylabel("Error Term (ε)")
        ax.set_title(f"Correlation between X and ε: {np.corrcoef(x, epsilon)[0, 1]:.3f}")
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

        # Display regression results
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Regression Results Comparison")
        st.markdown(f"""
        **Naive OLS (ignoring endogeneity)**:
        - β₁ estimate: {model.params[1]:.4f}
        - Bias: {model.params[1] - 3.0:.4f}
        - Standard error: {model.bse[1]:.4f}
        - t-statistic: {model.tvalues[1]:.4f}
        - p-value: {model.pvalues[1]:.4f}

        **Correct Model (including unobserved variable)**:
        - β₁ estimate: {model_correct.params[1]:.4f}
        - Bias: {model_correct.params[1] - 3.0:.4f}
        - Standard error: {model_correct.bse[1]:.4f}
        - Coefficient on unobserved variable: {model_correct.params[2]:.4f}

        **True β₁**: 3.0000
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # Monte Carlo simulation
    st.markdown('<div class="section-header">Monte Carlo Simulation: Impact of Endogeneity</div>',
                unsafe_allow_html=True)

    if st.button("Run Monte Carlo Simulation for Endogeneity", key="endogeneity_mc_button"):
        with st.spinner("Running simulation..."):
            # Function to generate exogenous data
            def generate_exogenous_data(n=100, beta0=2, beta1=3, error_sd=1, seed=42):
                np.random.seed(seed)
                x = np.random.uniform(0, 10, n)
                epsilon = np.random.normal(0, error_sd, n)
                y = beta0 + beta1 * x + epsilon
                return x, y, epsilon


            # Run simulations with different endogeneity levels
            endogeneity_levels = [0.0, 0.3, 0.6, 0.9]
            results = {}

            for endo_level in endogeneity_levels:
                betas = []
                t_stats = []

                # Store for instrumental variable approach
                iv_betas = []

                for i in range(1000):
                    if endo_level == 0:
                        # No endogeneity case
                        x, y, _ = generate_exogenous_data(n=n_samples, seed=i)
                    else:
                        # With endogeneity
                        x, y, _, unobserved = generate_endogenous_data(n=n_samples, endogeneity=endo_level, seed=i)

                    # OLS estimation
                    X = sm.add_constant(x)
                    model = sm.OLS(y, X).fit()

                    betas.append(model.params[1])
                    t_stats.append(model.tvalues[1])

                    # IV approach (if we had a valid instrument)
                    if endo_level > 0:
                        # Create a synthetic instrument (this is just for demonstration)
                        # In practice, finding a good instrument is difficult
                        np.random.seed(i + 10000)  # Different seed
                        instrument = np.random.normal(0, 1, n_samples) + 0.7 * x - 0.7 * unobserved

                        # First stage
                        X_first = sm.add_constant(instrument)
                        model_first = sm.OLS(x, X_first).fit()
                        x_hat = model_first.predict()

                        # Second stage
                        X_second = sm.add_constant(x_hat)
                        model_second = sm.OLS(y, X_second).fit()

                        iv_betas.append(model_second.params[1])

                results[endo_level] = {
                    "betas": np.array(betas),
                    "t_stats": np.array(t_stats)
                }

                if endo_level > 0:
                    results[endo_level]["iv_betas"] = np.array(iv_betas)

            col1, col2 = st.columns([1, 1])

            with col1:
                # Compare distribution of coefficient estimates
                fig, ax = plt.subplots(figsize=(10, 6))
                for level in endogeneity_levels:
                    sns.kdeplot(results[level]["betas"], ax=ax, label=f"Endogeneity = {level}")

                ax.axvline(3, color='black', linestyle='--', alpha=0.7, label="True β₁ = 3")
                ax.set_xlabel("Estimated β₁")
                ax.set_title("Sampling Distribution of β₁ for Different Endogeneity Levels")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)

                # Plot relationship between endogeneity level and bias
                mean_betas = [np.mean(results[level]["betas"]) for level in endogeneity_levels]
                biases = [mean_beta - 3 for mean_beta in mean_betas]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(endogeneity_levels, biases, marker='o', linestyle='-', linewidth=2)
                ax.axhline(0, color='red', linestyle='--', alpha=0.7)
                ax.set_xlabel("Endogeneity Level")
                ax.set_ylabel("Bias in β₁")
                ax.set_title("Relationship Between Endogeneity and Bias")
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)

            with col2:
                # Compare distribution of t-statistics
                fig, ax = plt.subplots(figsize=(10, 6))
                for level in endogeneity_levels:
                    sns.kdeplot(results[level]["t_stats"], ax=ax, label=f"Endogeneity = {level}")

                # Add critical values
                crit_val = stats.t.ppf(0.975, n_samples - 2)
                ax.axvline(-crit_val, color='red', linestyle='--', alpha=0.7)
                ax.axvline(crit_val, color='red', linestyle='--', alpha=0.7)

                ax.set_xlabel("t-statistic for β₁")
                ax.set_title("Distribution of t-statistics for Different Endogeneity Levels")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)

                # Compare OLS and IV for highest endogeneity level
                if len(endogeneity_levels) > 1 and endogeneity_levels[-1] > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    highest_level = endogeneity_levels[-1]

                    sns.kdeplot(results[highest_level]["betas"], ax=ax,
                                label=f"OLS (Endogeneity = {highest_level})")
                    sns.kdeplot(results[highest_level]["iv_betas"], ax=ax,
                                label=f"IV Estimator")
                    ax.axvline(3, color='black', linestyle='--', alpha=0.7,
                               label="True β₁ = 3")

                    ax.set_xlabel("Estimated β₁")
                    ax.set_title("OLS vs. IV Estimator with High Endogeneity")
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    st.pyplot(fig)

            # Create summary table
            summary_data = []
            for level in endogeneity_levels:
                # OLS statistics
                mean_beta = np.mean(results[level]["betas"])
                bias = mean_beta - 3
                std_beta = np.std(results[level]["betas"])
                mse = np.mean((results[level]["betas"] - 3) ** 2)

                # Rejection rates (for H0: β1 = 3)
                t_stats_centered = (results[level]["betas"] - 3) / (std_beta / np.sqrt(n_samples))
                rejection_rate = np.mean(np.abs(t_stats_centered) > crit_val)

                summary_data.append({
                    "Endogeneity": level,
                    "Mean β₁": mean_beta,
                    "Bias": bias,
                    "SD of β₁": std_beta,
                    "MSE": mse,
                    "Type I Error Rate": rejection_rate
                })

                # Add IV estimator results
                if level > 0:
                    mean_iv_beta = np.mean(results[level]["iv_betas"])
                    bias_iv = mean_iv_beta - 3
                    std_iv_beta = np.std(results[level]["iv_betas"])
                    mse_iv = np.mean((results[level]["iv_betas"] - 3) ** 2)

                    summary_data.append({
                        "Endogeneity": f"{level} (IV)",
                        "Mean β₁": mean_iv_beta,
                        "Bias": bias_iv,
                        "SD of β₁": std_iv_beta,
                        "MSE": mse_iv,
                        "Type I Error Rate": np.nan
                    })

            summary_df = pd.DataFrame(summary_data).set_index("Endogeneity")
            summary_df = summary_df.round(4)

            st.markdown('<div class="important-note">', unsafe_allow_html=True)
            st.markdown("""
            **Key Findings**:

            1. **Biased Estimates**: Endogeneity introduces systematic bias that doesn't diminish with sample size
            2. **Direction of Bias**: In this example, the bias is negative (estimates lower than true value)
            3. **Invalid Inference**: t-statistics and p-values become meaningless
            4. **Rejection Rates**: High rejection rates for true null hypotheses (high Type I error)
            5. **Instrumental Variables**: A potential solution, though instruments are often hard to find in practice
            """)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### Summary Statistics from Simulation")
            st.table(summary_df.T)

    st.markdown('<div class="section-header">Sources of Endogeneity and Solutions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Common Sources of Endogeneity")
        st.markdown("""
        1. **Omitted Variables**:
           - Important variables missing from the model
           - Confounding factors affecting both X and Y

        2. **Measurement Error**:
           - Inaccurately measured independent variables
           - Classical measurement error biases coefficients toward zero

        3. **Simultaneity / Reverse Causality**:
           - X causes Y and Y causes X
           - Common in economic relationships (e.g., supply and demand)

        4. **Sample Selection Bias**:
           - Non-random selection into the sample
           - Self-selection into treatment

        5. **Dynamics with Lagged Dependent Variables**:
           - Including lagged Y with autocorrelated errors
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Solutions to Endogeneity")
        st.markdown("""
        1. **Instrumental Variables (IV)**:
           - Two-Stage Least Squares (2SLS)
           - Requires valid instruments (relevant and exogenous)

        2. **Natural Experiments**:
           - Exogenous variation from policy changes, etc.
           - Difference-in-Differences
           - Regression Discontinuity

        3. **Panel Data Methods**:
           - Fixed effects to control for time-invariant confounders
           - First-differencing
           - Arellano-Bond estimator

        4. **Control Function Approaches**:
           - Heckman correction for sample selection
           - Proxy variables

        5. **Structural Equation Modeling**:
           - Explicitly model the simultaneous relationships
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Comparative Analysis
elif page == "Comparative Analysis":
    st.markdown('<div class="sub-header">Comparative Analysis of OLS Assumption Violations</div>',
                unsafe_allow_html=True)

    st.markdown("""
    Let's compare how different violations of OLS assumptions affect the performance of the estimator and 
    statistical inference. This will help students understand which violations are most concerning and why.
    """)

    # Table comparing the effects of different violations
    st.markdown('<div class="section-header">Comparison Table of Assumption Violations</div>', unsafe_allow_html=True)

    comparison_data = {
        "Violation": ["Nonlinearity", "Non-normality", "Heteroscedasticity", "Autocorrelation", "Multicollinearity",
                      "Endogeneity"],
        "Biased?": ["Yes", "No", "No", "No", "No", "Yes"],
        "Consistent?": ["No", "Yes", "Yes", "Yes", "Yes", "No"],
        "Efficient?": ["No", "Maybe*", "No", "No", "No", "No"],
        "Standard Errors": ["Incorrect", "Approximately correct (large n)", "Incorrect (usually underestimated)",
                            "Incorrect (usually underestimated)", "Inflated", "Incorrect"],
        "t/F Statistics": ["Invalid", "Approximately valid (large n)", "Invalid", "Invalid", "Reduced power",
                           "Invalid"],
        "Severity": ["High", "Low to Medium", "Medium", "Medium", "Medium", "Very High"],
        "Worsens with Large n?": ["Yes", "No", "No", "No", "No", "Yes"]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df.set_index("Violation"))

    st.markdown("""
    *For non-normality: Efficiency depends on the specific error distribution. OLS is most efficient when errors are normal.
    """)

    # Visual comparison of biases
    st.markdown('<div class="section-header">Visual Comparison of Biases</div>', unsafe_allow_html=True)

    if st.button("Generate Comparison of Biases", key="bias_comparison_button"):
        with st.spinner("Generating comparison plot..."):
            # Number of simulations
            n_sim = 1000
            n_samples = 100

            # Dictionary to store all results
            all_results = {}

            # Linear (correct) model
            np.random.seed(42)
            betas_correct = []
            for i in range(n_sim):
                x, y, _ = generate_linear_data(n=n_samples, seed=i)
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                betas_correct.append(model.params[1])
            all_results["Correct Model"] = np.array(betas_correct)

            # Nonlinear misspecification
            np.random.seed(42)
            betas_nonlinear = []
            for i in range(n_sim):
                x, y, _ = generate_nonlinear_data(n=n_samples, beta2=0.5, seed=i)
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                betas_nonlinear.append(model.params[1])
            all_results["Nonlinearity"] = np.array(betas_nonlinear)

            # Non-normal errors (t-distributed)
            np.random.seed(42)
            betas_nonnormal = []
            for i in range(n_sim):
                np.random.seed(i)
                x = np.random.uniform(0, 10, n_samples)
                epsilon = np.random.standard_t(3, n_samples)
                y = 2 + 3 * x + epsilon
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                betas_nonnormal.append(model.params[1])
            all_results["Non-normality"] = np.array(betas_nonnormal)

            # Heteroscedasticity
            np.random.seed(42)
            betas_hetero = []
            for i in range(n_sim):
                x, y, _ = generate_heteroscedastic_data(n=n_samples, seed=i)
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                betas_hetero.append(model.params[1])
            all_results["Heteroscedasticity"] = np.array(betas_hetero)

            # Autocorrelation
            np.random.seed(42)
            betas_autocorr = []
            for i in range(n_sim):
                x, y, _ = generate_autocorrelated_data(n=n_samples, rho=0.7, seed=i)
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                betas_autocorr.append(model.params[1])
            all_results["Autocorrelation"] = np.array(betas_autocorr)

            # Multicollinearity
            np.random.seed(42)
            betas_multicol = []
            for i in range(n_sim):
                x1, x2, y, _ = generate_multicollinear_data(n=n_samples, correlation=0.9, seed=i)
                X = sm.add_constant(np.column_stack((x1, x2)))
                model = sm.OLS(y, X).fit()
                betas_multicol.append(model.params[1])  # Focus on beta1
            all_results["Multicollinearity"] = np.array(betas_multicol)

            # Endogeneity
            np.random.seed(42)
            betas_endo = []
            for i in range(n_sim):
                x, y, _, _ = generate_endogenous_data(n=n_samples, endogeneity=0.7, seed=i)
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                betas_endo.append(model.params[1])
            all_results["Endogeneity"] = np.array(betas_endo)

            # Create violin plot
            fig, ax = plt.subplots(figsize=(14, 8))

            # Collect all data for the violin plot
            violin_data = []
            labels = []

            # Plot in order of increasing bias
            order = ["Correct Model", "Non-normality", "Heteroscedasticity", "Autocorrelation",
                     "Multicollinearity", "Nonlinearity", "Endogeneity"]

            for label in order:
                violin_data.append(all_results[label])
                labels.append(label)

            # Create violin plot
            parts = ax.violinplot(violin_data, showmeans=True, showmedians=False)

            # Color the violins
            colors = ['green', 'blue', 'blue', 'blue', 'orange', 'red', 'red']
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)

            # Add true value line
            ax.axhline(y=3, color='black', linestyle='--', alpha=0.7, label="True β₁ = 3")

            # Mean markers
            means = [np.mean(all_results[label]) for label in order]
            ax.scatter(range(1, len(labels) + 1), means, marker='o', color='white', s=30, zorder=3)

            # Customize the plot
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel("Estimated β₁")
            ax.set_title("Distribution of β₁ Estimates Under Different Assumption Violations")
            ax.grid(True, linestyle='--', alpha=0.7)

            # Add annotations for mean values
            for i, mean in enumerate(means):
                ax.annotate(f"{mean:.3f}",
                            xy=(i + 1, mean),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center',
                            va='bottom',
                            fontweight='bold')

            # Legend
            import matplotlib.patches as mpatches

            green_patch = mpatches.Patch(color='green', alpha=0.7, label='Unbiased, Efficient')
            blue_patch = mpatches.Patch(color='blue', alpha=0.7, label='Unbiased, Inefficient')
            orange_patch = mpatches.Patch(color='orange', alpha=0.7, label='Unbiased, High Variance')
            red_patch = mpatches.Patch(color='red', alpha=0.7, label='Biased')
            true_line = mpatches.Patch(color='black', alpha=0.7, label='True β₁ = 3')

            ax.legend(handles=[green_patch, blue_patch, orange_patch, red_patch, true_line],
                      loc='upper right', framealpha=0.9)

            plt.tight_layout()
            st.pyplot(fig)

    # Standard error comparison
    st.markdown('<div class="section-header">Standard Error Comparison</div>', unsafe_allow_html=True)

    if st.button("Generate Standard Error Comparison", key="se_comparison_button"):
        with st.spinner("Generating standard error comparison..."):
            # Number of simulations
            n_sim = 500
            n_samples = 100

            # Dictionary to store results
            se_results = {
                "Model": [],
                "Reported SE": [],
                "Empirical SE": [],
                "True Coverage (95%)": []
            }

            # Define models to evaluate
            models = ["Correct Model", "Non-normality", "Heteroscedasticity", "Autocorrelation",
                      "Multicollinearity", "Nonlinearity", "Endogeneity"]

            # Loop through models
            for model_name in models:
                reported_se = []
                estimated_betas = []

                # Run simulations for each model
                for i in range(n_sim):
                    # Generate data based on the model
                    if model_name == "Correct Model":
                        x, y, _ = generate_linear_data(n=n_samples, seed=i)
                        X = sm.add_constant(x)
                        model = sm.OLS(y, X).fit()
                        reported_se.append(model.bse[1])
                        estimated_betas.append(model.params[1])

                    elif model_name == "Non-normality":
                        np.random.seed(i)
                        x = np.random.uniform(0, 10, n_samples)
                        epsilon = np.random.standard_t(3, n_samples)
                        y = 2 + 3 * x + epsilon
                        X = sm.add_constant(x)
                        model = sm.OLS(y, X).fit()
                        reported_se.append(model.bse[1])
                        estimated_betas.append(model.params[1])

                    elif model_name == "Heteroscedasticity":
                        x, y, _ = generate_heteroscedastic_data(n=n_samples, seed=i)
                        X = sm.add_constant(x)
                        model = sm.OLS(y, X).fit()
                        reported_se.append(model.bse[1])
                        estimated_betas.append(model.params[1])

                    elif model_name == "Autocorrelation":
                        x, y, _ = generate_autocorrelated_data(n=n_samples, rho=0.7, seed=i)
                        X = sm.add_constant(x)
                        model = sm.OLS(y, X).fit()
                        reported_se.append(model.bse[1])
                        estimated_betas.append(model.params[1])

                    elif model_name == "Multicollinearity":
                        x1, x2, y, _ = generate_multicollinear_data(n=n_samples, correlation=0.9, seed=i)
                        X = sm.add_constant(np.column_stack((x1, x2)))
                        model = sm.OLS(y, X).fit()
                        reported_se.append(model.bse[1])
                        estimated_betas.append(model.params[1])

                    elif model_name == "Nonlinearity":
                        x, y, _ = generate_nonlinear_data(n=n_samples, beta2=0.5, seed=i)
                        X = sm.add_constant(x)
                        model = sm.OLS(y, X).fit()
                        reported_se.append(model.bse[1])
                        estimated_betas.append(model.params[1])

                    elif model_name == "Endogeneity":
                        x, y, _, _ = generate_endogenous_data(n=n_samples, endogeneity=0.7, seed=i)
                        X = sm.add_constant(x)
                        model = sm.OLS(y, X).fit()
                        reported_se.append(model.bse[1])
                        estimated_betas.append(model.params[1])

                # Calculate empirical standard error
                empirical_se = np.std(estimated_betas)

                # Calculate true coverage
                contains_true = []
                for b, se in zip(estimated_betas, reported_se):
                    lower = b - 1.96 * se
                    upper = b + 1.96 * se
                    contains_true.append(lower <= 3 <= upper)

                coverage = np.mean(contains_true) * 100

                # Add to results
                se_results["Model"].append(model_name)
                se_results["Reported SE"].append(np.mean(reported_se))
                se_results["Empirical SE"].append(empirical_se)
                se_results["True Coverage (95%)"].append(f"{coverage:.1f}%")

            # Create DataFrame
            se_df = pd.DataFrame(se_results)

            # Calculate ratio (to show over/under estimation)
            se_df["Reported/Actual Ratio"] = se_df["Reported SE"] / se_df["Empirical SE"]

            # Reorder columns
            se_df = se_df[["Model", "Reported SE", "Empirical SE", "Reported/Actual Ratio", "True Coverage (95%)"]]

            # Display table
            st.table(se_df.set_index("Model"))

            # Create a bar chart comparing reported vs empirical SE
            fig, ax = plt.subplots(figsize=(14, 8))

            models = se_df["Model"].tolist()
            x = np.arange(len(models))
            width = 0.35

            # Bars
            reported_bars = ax.bar(x - width / 2, se_df["Reported SE"], width, label="Average Reported SE",
                                   color="blue", alpha=0.7)
            empirical_bars = ax.bar(x + width / 2, se_df["Empirical SE"], width, label="Empirical SE", color="red",
                                    alpha=0.7)

            # Labels and customization
            ax.set_ylabel("Standard Error")
            ax.set_title("Comparison of Reported vs. Actual Standard Errors")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.legend()


            # Add value annotations
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f"{height:.3f}",
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha="center", va="bottom", fontsize=9)


            autolabel(reported_bars)
            autolabel(empirical_bars)

            plt.tight_layout()
            st.pyplot(fig)

            # Add a coverage plot
            fig, ax = plt.subplots(figsize=(14, 6))

            # Extract coverage percentages
            coverage_values = [float(x.strip('%')) for x in se_df["True Coverage (95%)"].tolist()]

            # Create bar chart
            bars = ax.bar(models, coverage_values, color='teal', alpha=0.7)

            # Add reference line for 95%
            ax.axhline(y=95, color='red', linestyle='--', label="Nominal 95% Level")

            # Labels and customization
            ax.set_ylabel("Actual Coverage (%)")
            ax.set_title("True Coverage of 95% Confidence Intervals")
            ax.set_ylim(min(70, min(coverage_values) - 5), 100)
            plt.xticks(rotation=45, ha="right")
            ax.legend()

            # Add value annotations
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.annotate(f"{height:.1f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig)

    # Key takeaways
    st.markdown('<div class="section-header">Key Takeaways</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Severity of Assumption Violations

    **Most Critical Violations (Affecting Bias):**
    - **Endogeneity**: Causes bias that does not disappear with large samples. Results in inconsistent estimators.
    - **Nonlinearity (Model Misspecification)**: Estimating a linear model when the true relationship is nonlinear leads to biased estimates.

    **Moderate Violations (Affecting Efficiency and Inference):**
    - **Heteroscedasticity**: Estimators remain unbiased but standard errors are incorrect, leading to invalid hypothesis tests.
    - **Autocorrelation**: Similar to heteroscedasticity, affects inference rather than central estimates.
    - **Multicollinearity**: Results in larger standard errors, making it harder to achieve statistical significance.

    **Least Critical Violation:**
    - **Non-normality**: With large samples, the central limit theorem ensures that inference remains approximately valid.

    ### Diagnostic Priorities

    Based on this analysis, students should prioritize testing for:
    1. Endogeneity (though this is often the hardest to detect)
    2. Model misspecification/nonlinearity
    3. Heteroscedasticity and autocorrelation
    4. Multicollinearity (when working with multiple predictors)
    5. Non-normality (least concerning with reasonably large samples)

    ### Remedial Actions

    When violations are detected, different corrections are needed:
    - For endogeneity: Instrumental variables or natural experiments
    - For nonlinearity: Transform variables or use more flexible functional forms
    - For heteroscedasticity/autocorrelation: Use robust standard errors
    - For multicollinearity: Collect more data, use regularization techniques, or create composite variables
    - For non-normality: Larger samples generally solve the problem, or bootstrap inference
    """)

    # Interactive example selector
    st.markdown('<div class="section-header">Interactive Assumption Violation Simulator</div>', unsafe_allow_html=True)

    st.markdown("""
    The simulator below lets you explore how different violations affect estimation and inference simultaneously.
    Select a violation type and adjust its severity to see the impact on parameter estimates and confidence intervals.
    """)

    # Create selection widgets
    col1, col2 = st.columns(2)

    with col1:
        violation_type = st.selectbox("Select Violation Type",
                                      ["Nonlinearity", "Non-normality", "Heteroscedasticity",
                                       "Autocorrelation", "Multicollinearity", "Endogeneity"])

    with col2:
        severity = st.slider("Violation Severity", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    # Generate simulation based on selection
    if st.button("Run Simulation", key="interactive_simulation"):
        with st.spinner("Running simulation..."):
            # Generate appropriate dataset based on selection
            n_samples = 200
            n_sim = 100

            if violation_type == "Nonlinearity":
                # For nonlinearity, severity controls the coefficient of the squared term
                beta2 = severity * 1.0  # Max value of 1.0 for squared term
                x, y, true_curve = generate_nonlinear_data(n=n_samples, beta2=beta2, seed=42)

                # Fit linear model
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()

                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot data points
                ax.scatter(x, y, alpha=0.6, label="Data points")

                # Sort x for smooth plotting
                x_sorted = np.sort(x)

                # Plot true curve
                y_true = 2 + 3 * x_sorted + beta2 * x_sorted ** 2
                ax.plot(x_sorted, y_true, 'r-', linewidth=2, label="True relationship")

                # Plot fitted line
                y_pred = model.params[0] + model.params[1] * x_sorted
                ax.plot(x_sorted, y_pred, 'g-', linewidth=2, label="OLS fit")

                # Customize
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(f"Nonlinearity Example (Quadratic Term = {beta2:.1f})")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Show plot
                st.pyplot(fig)

                # Run multiple simulations to show parameter distribution
                betas = []
                for i in range(n_sim):
                    x, y, _ = generate_nonlinear_data(n=n_samples, beta2=beta2, seed=i)
                    X = sm.add_constant(x)
                    model = sm.OLS(y, X).fit()
                    betas.append(model.params[1])

                # Display parameter statistics
                st.markdown(f"### Statistics from {n_sim} Simulations")
                st.markdown(f"True β₁ = 3")
                st.markdown(f"Average estimated β₁ = {np.mean(betas):.4f}")
                st.markdown(f"Bias = {np.mean(betas) - 3:.4f}")

                # Plot histogram of estimates
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(betas, bins=20, alpha=0.7, color='blue')
                ax.axvline(x=3, color='red', linestyle='--', label="True β₁")
                ax.axvline(x=np.mean(betas), color='green', linestyle='-', label="Mean estimate")
                ax.set_xlabel("Estimated β₁")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Distribution of β₁ Estimates with Nonlinearity (β₂ = {beta2:.1f})")
                ax.legend()
                st.pyplot(fig)

            elif violation_type == "Non-normality":
                # For non-normality, severity controls the degrees of freedom in t-distribution
                # Lower df = heavier tails
                df = max(1, int(10 * (1 - severity) + 1))  # Maps severity 0->10 df, 1->1 df

                # Generate data
                np.random.seed(42)
                x = np.random.uniform(0, 10, n_samples)
                epsilon = np.random.standard_t(df, n_samples)
                y = 2 + 3 * x + epsilon

                # Fit model
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()

                # Create QQ plot for residuals
                fig, ax = plt.subplots(figsize=(10, 6))
                sm.qqplot(model.resid, line='45', ax=ax)
                ax.set_title(f"QQ Plot of Residuals (t-distribution with df={df})")
                st.pyplot(fig)

                # Display results table
                st.write(f"### OLS Results with Non-normal Errors (t-distribution, df={df})")
                st.write(f"True β₁ = 3")
                st.write(f"Estimated β₁ = {model.params[1]:.4f}")
                st.write(f"Standard Error = {model.bse[1]:.4f}")
                st.write(f"95% CI: [{model.conf_int()[1][0]:.4f}, {model.conf_int()[1][1]:.4f}]")

                # Run multiple simulations
                betas = []
                ses = []
                contains_true = []

                for i in range(n_sim):
                    np.random.seed(i)
                    x = np.random.uniform(0, 10, n_samples)
                    epsilon = np.random.standard_t(df, n_samples)
                    y = 2 + 3 * x + epsilon

                    X = sm.add_constant(x)
                    model = sm.OLS(y, X).fit()

                    betas.append(model.params[1])
                    ses.append(model.bse[1])

                    lower = model.params[1] - 1.96 * model.bse[1]
                    upper = model.params[1] + 1.96 * model.bse[1]
                    contains_true.append(lower <= 3 <= upper)

                # Display parameter distribution statistics
                st.markdown(f"### Statistics from {n_sim} Simulations")
                st.markdown(f"Average β₁ = {np.mean(betas):.4f}")
                st.markdown(f"True coverage rate of 95% CI: {np.mean(contains_true) * 100:.1f}%")

            elif violation_type == "Heteroscedasticity":
                # Generate data where variance increases with x
                # Severity controls how strongly variance depends on x
                x, y, _ = generate_heteroscedastic_data(n=n_samples, severity=severity, seed=42)

                # Fit model
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()

                # Also fit with robust standard errors
                robust_model = sm.OLS(y, X).fit(cov_type='HC3')

                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot data points
                ax.scatter(x, y, alpha=0.6)

                # Sort x for smooth plotting
                x_sorted = np.sort(x)
                X_sorted = sm.add_constant(x_sorted)

                # Plot fitted line
                y_pred = model.predict(X_sorted)
                ax.plot(x_sorted, y_pred, 'r-', linewidth=2, label="OLS fit")

                # Add 95% prediction intervals
                se_pred = model.get_prediction(X_sorted).se_mean
                ax.fill_between(x_sorted, y_pred - 1.96 * se_pred, y_pred + 1.96 * se_pred,
                                alpha=0.2, color='blue', label="95% CI (standard)")

                # Add robust 95% prediction intervals using HC3 errors
                se_robust = robust_model.get_prediction(X_sorted).se_mean
                ax.fill_between(x_sorted, y_pred - 1.96 * se_robust, y_pred + 1.96 * se_robust,
                                alpha=0.2, color='red', label="95% CI (robust)")

                # Customize
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(f"Heteroscedasticity Example (Severity = {severity:.1f})")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Show plot
                st.pyplot(fig)

                # Show comparison of standard errors
                st.markdown("### Comparing Standard and Robust Standard Errors")

                # Create dataframe for display
                se_df = pd.DataFrame({
                    "": ["Standard", "Robust (HC3)"],
                    "β₁ Estimate": [model.params[1], robust_model.params[1]],
                    "Standard Error": [model.bse[1], robust_model.bse[1]],
                    "95% CI Lower": [model.conf_int()[1][0], robust_model.conf_int()[1][0]],
                    "95% CI Upper": [model.conf_int()[1][1], robust_model.conf_int()[1][1]],
                    "t-statistic": [model.tvalues[1], robust_model.tvalues[1]],
                    "p-value": [model.pvalues[1], robust_model.pvalues[1]]
                })

                st.table(se_df.set_index(""))

                # Run simulations to show true coverage
                std_ses = []
                robust_ses = []
                std_coverage = []
                robust_coverage = []

                for i in range(n_sim):
                    x, y, _ = generate_heteroscedastic_data(n=n_samples, severity=severity, seed=i)
                    X = sm.add_constant(x)

                    # Standard model
                    std_model = sm.OLS(y, X).fit()
                    std_ses.append(std_model.bse[1])

                    # Check if CI contains true value
                    lower = std_model.params[1] - 1.96 * std_model.bse[1]
                    upper = std_model.params[1] + 1.96 * std_model.bse[1]
                    std_coverage.append(lower <= 3 <= upper)

                    # Robust model
                    robust_model = sm.OLS(y, X).fit(cov_type='HC3')
                    robust_ses.append(robust_model.bse[1])

                    # Check if robust CI contains true value
                    lower = robust_model.params[1] - 1.96 * robust_model.bse[1]
                    upper = robust_model.params[1] + 1.96 * robust_model.bse[1]
                    robust_coverage.append(lower <= 3 <= upper)

                # Display coverage statistics
                st.markdown(f"### Coverage Statistics from {n_sim} Simulations")
                st.markdown(f"True coverage of standard 95% CI: {np.mean(std_coverage) * 100:.1f}%")
                st.markdown(f"True coverage of robust 95% CI: {np.mean(robust_coverage) * 100:.1f}%")

            elif violation_type == "Autocorrelation":
                # Generate autocorrelated data
                # Severity controls the correlation coefficient rho
                rho = severity * 0.95  # Max value of 0.95
                x, y, _ = generate_autocorrelated_data(n=n_samples, rho=rho, seed=42)

                # Fit standard model
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()

                # Also fit with HAC standard errors (Newey-West)
                hac_model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

                # Plot residuals over time to show autocorrelation
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(model.resid, marker='o', linestyle='-', markersize=4, alpha=0.7)
                ax.set_xlabel("Observation Number")
                ax.set_ylabel("Residual")
                ax.set_title(f"Residuals Over Time (ρ = {rho:.2f})")
                ax.grid(True, alpha=0.3)

                # Show autocorrelation plot
                st.pyplot(fig)

                # Calculate and plot autocorrelation function
                from statsmodels.graphics.tsaplots import plot_acf

                fig, ax = plt.subplots(figsize=(10, 6))
                plot_acf(model.resid, lags=20, ax=ax)
                ax.set_title(f"Autocorrelation Function of Residuals (ρ = {rho:.2f})")

                st.pyplot(fig)

                # Show comparison of standard errors
                st.markdown("### Comparing Standard and HAC Standard Errors")

                # Create dataframe for display
                se_df = pd.DataFrame({
                    "": ["Standard", "HAC (Newey-West)"],
                    "β₁ Estimate": [model.params[1], hac_model.params[1]],
                    "Standard Error": [model.bse[1], hac_model.bse[1]],
                    "95% CI Lower": [model.conf_int()[1][0], hac_model.conf_int()[1][0]],
                    "95% CI Upper": [model.conf_int()[1][1], hac_model.conf_int()[1][1]],
                    "t-statistic": [model.tvalues[1], hac_model.tvalues[1]],
                    "p-value": [model.pvalues[1], hac_model.pvalues[1]]
                })

                st.table(se_df.set_index(""))

                # Run simulations to calculate coverage
                std_coverage = []
                hac_coverage = []

                for i in range(n_sim):
                    x, y, _ = generate_autocorrelated_data(n=n_samples, rho=rho, seed=i)
                    X = sm.add_constant(x)

                    # Standard model
                    std_model = sm.OLS(y, X).fit()

                    # Check if CI contains true value
                    lower = std_model.params[1] - 1.96 * std_model.bse[1]
                    upper = std_model.params[1] + 1.96 * std_model.bse[1]
                    std_coverage.append(lower <= 3 <= upper)

                    # HAC model
                    hac_model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})

                    # Check if HAC CI contains true value
                    lower = hac_model.params[1] - 1.96 * hac_model.bse[1]
                    upper = hac_model.params[1] + 1.96 * hac_model.bse[1]
                    hac_coverage.append(lower <= 3 <= upper)

                # Display coverage statistics
                st.markdown(f"### Coverage Statistics from {n_sim} Simulations")
                st.markdown(f"True coverage of standard 95% CI: {np.mean(std_coverage) * 100:.1f}%")
                st.markdown(f"True coverage of HAC 95% CI: {np.mean(hac_coverage) * 100:.1f}%")

            elif violation_type == "Multicollinearity":
                # Generate multicollinear data
                # Severity controls correlation between predictors
                correlation = severity * 0.99  # Max correlation of 0.99
                x1, x2, y, _ = generate_multicollinear_data(n=n_samples, correlation=correlation, seed=42)

                # Fit model
                X = sm.add_constant(np.column_stack((x1, x2)))
                model = sm.OLS(y, X).fit()

                # Create scatter plot to show correlation
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(x1, x2, alpha=0.6)
                ax.set_xlabel("X1")
                ax.set_ylabel("X2")
                ax.set_title(f"Correlation Between Predictors (r = {correlation:.2f})")

                # Add correlation line
                z = np.polyfit(x1, x2, 1)
                p = np.poly1d(z)
                x1_sorted = np.sort(x1)
                ax.plot(x1_sorted, p(x1_sorted), "r--", alpha=0.8)

                # Add correlation text
                ax.text(0.05, 0.95, f"Correlation = {correlation:.3f}", transform=ax.transAxes,
                        fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Highlight VIF
                from statsmodels.stats.outliers_influence import variance_inflation_factor

                vif1 = variance_inflation_factor(X, 1)
                vif2 = variance_inflation_factor(X, 2)

                st.markdown("### Variance Inflation Factors (VIF)")
                st.markdown(f"VIF for X1: {vif1:.2f}")
                st.markdown(f"VIF for X2: {vif2:.2f}")
                st.markdown("*VIF > 10 typically indicates problematic multicollinearity*")

                # Display regression results
                st.markdown("### Regression Results")

                # Create summary table
                results_df = pd.DataFrame({
                    "": ["Intercept", "β₁ (X1)", "β₂ (X2)"],
                    "Coefficient": [model.params[0], model.params[1], model.params[2]],
                    "Standard Error": [model.bse[0], model.bse[1], model.bse[2]],
                    "t-statistic": [model.tvalues[0], model.tvalues[1], model.tvalues[2]],
                    "p-value": [model.pvalues[0], model.pvalues[1], model.pvalues[2]]
                })

                st.table(results_df.set_index(""))

                # Run simulations to show variability in estimates
                beta1_estimates = []
                beta2_estimates = []
                se1_estimates = []
                se2_estimates = []

                for i in range(n_sim):
                    x1, x2, y, _ = generate_multicollinear_data(n=n_samples, correlation=correlation, seed=i)
                    X = sm.add_constant(np.column_stack((x1, x2)))
                    model = sm.OLS(y, X).fit()

                    beta1_estimates.append(model.params[1])
                    beta2_estimates.append(model.params[2])
                    se1_estimates.append(model.bse[1])
                    se2_estimates.append(model.bse[2])

                # Create violin plots for beta distributions
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Beta1 distribution
                ax1.violinplot(beta1_estimates, showmeans=True)
                ax1.axhline(y=2, color='red', linestyle='--', label="True β₁ = 2")
                ax1.set_title(f"Distribution of β₁ Estimates")
                ax1.set_ylabel("Estimated Coefficient")
                ax1.set_xticks([1])
                ax1.set_xticklabels(["β₁"])
                ax1.legend()

                # Beta2 distribution
                ax2.violinplot(beta2_estimates, showmeans=True)
                ax2.axhline(y=3, color='red', linestyle='--', label="True β₂ = 3")
                ax2.set_title(f"Distribution of β₂ Estimates")
                ax2.set_ylabel("Estimated Coefficient")
                ax2.set_xticks([1])
                ax2.set_xticklabels(["β₂"])
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig)

                # Display simulation statistics
                st.markdown(f"### Statistics from {n_sim} Simulations")
                st.markdown(f"β₁ standard deviation: {np.std(beta1_estimates):.4f}")
                st.markdown(f"β₂ standard deviation: {np.std(beta2_estimates):.4f}")
                st.markdown(f"Average β₁ SE: {np.mean(se1_estimates):.4f}")
                st.markdown(f"Average β₂ SE: {np.mean(se2_estimates):.4f}")

            elif violation_type == "Endogeneity":
                # Generate endogenous data
                # Severity controls strength of endogeneity
                endogeneity = severity * 0.95  # Max correlation of 0.95
                x, y, z, true_beta = generate_endogenous_data(n=n_samples, endogeneity=endogeneity, seed=42)

                # Fit OLS model
                X = sm.add_constant(x)
                ols_model = sm.OLS(y, X).fit()

                # Fit IV model using z as instrument
                iv_model = sm.IV2SLS(y, X, instruments=sm.add_constant(z)).fit()

                # Create scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot data points
                ax.scatter(x, y, alpha=0.5, label="Data points")

                # Sort x for smooth plotting
                x_sorted_idx = np.argsort(x)
                x_sorted = x[x_sorted_idx]
                X_sorted = sm.add_constant(x_sorted)

                # Plot true relationship
                y_true = true_beta[0] + true_beta[1] * x_sorted
                ax.plot(x_sorted, y_true, 'g-', linewidth=2, label="True relationship")

                # Plot OLS fit
                y_ols = ols_model.predict(X_sorted)
                ax.plot(x_sorted, y_ols, 'r-', linewidth=2, label="OLS fit")

                # Plot IV fit
                y_iv = iv_model.predict(X_sorted)
                ax.plot(x_sorted, y_iv, 'b--', linewidth=2, label="IV fit")

                # Customize
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(f"Endogeneity Example (Severity = {endogeneity:.2f})")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Show plot
                st.pyplot(fig)

                # Display regression results comparison
                st.markdown("### OLS vs. IV Regression Results")

                # Create comparison table
                comparison_df = pd.DataFrame({
                    "": ["True β₁", "OLS β₁", "IV β₁"],
                    "Estimate": [true_beta[1], ols_model.params[1], iv_model.params[1]],
                    "Bias": [0, ols_model.params[1] - true_beta[1], iv_model.params[1] - true_beta[1]],
                    "Standard Error": ["N/A", ols_model.bse[1], iv_model.bse[1]]
                })

                st.table(comparison_df.set_index(""))

                # Run simulations to compare OLS and IV
                ols_betas = []
                iv_betas = []

                for i in range(n_sim):
                    x, y, z, true_beta = generate_endogenous_data(n=n_samples, endogeneity=endogeneity, seed=i)
                    X = sm.add_constant(x)

                    # OLS
                    ols_model = sm.OLS(y, X).fit()
                    ols_betas.append(ols_model.params[1])

                    # IV
                    iv_model = sm.IV2SLS(y, X, instruments=sm.add_constant(z)).fit()
                    iv_betas.append(iv_model.params[1])

                # Create plot comparing distributions
                fig, ax = plt.subplots(figsize=(12, 6))

                # Create violin plots
                positions = [1, 2]
                v1 = ax.violinplot([ols_betas, iv_betas], positions=positions, showmeans=True)

                # Color violins
                v1['bodies'][0].set_facecolor('red')
                v1['bodies'][0].set_alpha(0.7)
                v1['bodies'][1].set_facecolor('blue')
                v1['bodies'][1].set_alpha(0.7)

                # True value line
                ax.axhline(y=true_beta[1], color='green', linestyle='--', label=f"True β₁ = {true_beta[1]}")

                # Add annotations for means
                ols_mean = np.mean(ols_betas)
                iv_mean = np.mean(iv_betas)

                ax.annotate(f"Mean: {ols_mean:.3f}", xy=(1, ols_mean), xytext=(10, 0),
                            textcoords="offset points", ha='left', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                ax.annotate(f"Mean: {iv_mean:.3f}", xy=(2, iv_mean), xytext=(10, 0),
                            textcoords="offset points", ha='left', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # Customize
                ax.set_xticks(positions)
                ax.set_xticklabels(['OLS', 'IV'])
                ax.set_ylabel("Estimated β₁")
                ax.set_title(f"Distribution of OLS vs. IV Estimates with Endogeneity (r = {endogeneity:.2f})")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Show plot
                st.pyplot(fig)

                # Display simulation statistics
                st.markdown(f"### Statistics from {n_sim} Simulations")
                st.markdown(f"True β₁ = {true_beta[1]}")
                st.markdown(f"Average OLS β₁ = {ols_mean:.4f} (Bias: {ols_mean - true_beta[1]:.4f})")
                st.markdown(f"Average IV β₁ = {iv_mean:.4f} (Bias: {iv_mean - true_beta[1]:.4f})")
                st.markdown("---")
                st.caption("Created by the Dr Merwan Roudane. Designed for educational purposes.")
