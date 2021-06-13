# Minimization of a function with Nelder-Mead Algorithm

Based on cosmological luminosity distance, this program find the best parameters of matter density, dark energy density and w from equation of state.  
It calculates the chi square between a fake dataset and tworetical result of the calculation of the distance modulus.  
The script find the best parameters by applying Nelder-Mead Algorithm on chi square function.  

## Integration using the Trapezoidal method
The C file find the chi square of the cosmological distance modulus, this function includes an integration. The script achieves it by finding the integral result with trapezoidal rule for integration.

## Example 1
![evolution_Nelder_Mead_fake_omMcte](mapping_chi2_fake_omMcte.png)

![evolution_params_fake_omMcte](evolution_params_fake_omMcte.gif)

![evolution_Nelder_Mead_fake_omMcte](evolution_Nelder_Mead_fake_omMcte.png)

## Example 2
![evolution_Nelder_Mead_fake_omMcte](mapping_chi2_fake_omEEcte.png)

![evolution_params_fake_omEEcte](evolution_params_fake_omEEcte.gif)

![evolution_Nelder_Mead_fake_omEEcte](evolution_Nelder_Mead_fake_omEEcte.png)

## Example 3
![evolution_Nelder_Mead_fake_omMcte](mapping_chi2_fake_omwcte.png)

![evolution_params_fake_wcte](evolution_params_fake_wcte.gif)

![evolution_Nelder_Mead_fake_wcte](evolution_Nelder_Mead_fake_wcte.png)