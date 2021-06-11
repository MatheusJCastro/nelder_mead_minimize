
#include<stdbool.h>
#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include<time.h>
#include<math.h>
#include<omp.h>

#define MAX 100
#define c 299792458 //velocidade da luz em m/s

double e_func(double z, double param[3], double omega_k){
    double omega_m_factor = param[0] * pow((1 + z), 3);
    double omega_de_factor = param[1] * pow((1 + z), (3 + 3 * param[2]));
    double omega_k_factor = omega_k * pow((1 + z), 2);

    double sq = 1/sqrt(omega_m_factor + omega_de_factor + omega_k_factor);
    return sq;
}

double calc_trapezium_formula(double lim[2], double n, double params[3], double omega_k){
    double h = (lim[1] - lim[0]) / n;
    double sum = 0;

    for(double j = lim[0] + h; j < lim[1]; j = j + 2*h){
        sum = sum + e_func(j, params, omega_k);
    }

    double trapezium = h * sum;
    return trapezium;
}

double integrate(double params[3], double omega_k, double lim[2], double eps_desired){
    if(lim[0] == lim[1])
        return 0;

    double result = (e_func(lim[0], params, omega_k) + e_func(lim[1], params, omega_k)) * (lim[1] - lim[0]) / 2;

    int count = 0;
    double eps = 1;
    
    while(eps >= eps_desired){
        count++;
        double result_old = result;
        double divisions = pow(2, count);

        result = result / 2 + calc_trapezium_formula(lim, divisions, params, omega_k);

        eps = fabs((result - result_old) / result);
    }

    return result;
}

double comoving_distance(double h0, double redshift, double params[3], double precision){
    double omega_k = 1 - (params[0] + params[1]);
    double hubble_distance = c / h0;
    double factor_k;

    double lim[2] = {0, redshift};
    double integration_result = integrate(params, omega_k, lim, precision);

    if(omega_k == 0){
        factor_k = hubble_distance * integrate(params, omega_k, lim, precision);
    }
    else if(omega_k > 0){
        double sqr_om = sqrt(omega_k);
        factor_k = hubble_distance / sqr_om * sinh(sqr_om * integration_result);
    }
    else{
        double sqr_om = sqrt(fabs(omega_k));
        factor_k = hubble_distance / sqr_om * sin(sqr_om * integration_result);
    }

    return factor_k;
}

double luminosity_distance(double h0, double redshift, double params[3], double precision){
    double lum = (1 + redshift) * comoving_distance(h0, redshift, params, precision);
    return lum;
}

double dist_mod(double dist_lum){
    //dist_lum is the luminosity distance in Mpc
    double mod = 5 * log10(dist_lum * pow(10, 6) / 10);
    return mod;
}

double lumin_dist_mod_func(double h0, double redshift, double params[3], double precision){
    double mpc_to_km = 3.086E+19; //conversão de Mpc para km

    h0 = h0 / mpc_to_km;

    double lum_dist_val = luminosity_distance(h0, redshift, params, precision);
    lum_dist_val = lum_dist_val * pow(10, -3) / mpc_to_km; //conversão de m para Mpc

    double dist_mod_val = dist_mod(lum_dist_val);

    return dist_mod_val;
}


double calc_chi(int h0, int nrows, double data[3][nrows], double params[3], double precision){
    double chi2 = 0;

    #pragma omp parallel shared(h0, nrows, data, params, precision)
    {
    	//if(omp_get_thread_num() == 0){
        //    printf("Executing for %d thread(s).\n", omp_get_num_threads());
        //}

    	#pragma omp for reduction(+: chi2)
        for(int i = 0; i < nrows; i++){
            double teor_data = lumin_dist_mod_func(h0, data[0][i], params, precision);
            chi2 += pow((data[1][i] - teor_data) / data[2][i], 2);
        }
    }

    return chi2;
}

double main_execution(char fl_name[MAX], double h0, double omega_m, double omega_ee, double w, double precision, int nrows){
    FILE *csv;
    char header[4][MAX];
    double data[3][nrows];

    csv = fopen(fl_name, "r");

    if(fscanf(csv, "%s %s %s %s\n", header[0], header[1], header[2], header[3])){}
    //printf("%s %s %s %s\n", header[0],header[1], header[2], header[3]);

    for (int i = 0; i < nrows ; i++) // Read until the last line.
    {
        if(fscanf(csv, "%lf %lf %lf\n", &data[0][i], &data[1][i], &data[2][i])){} // Create the matrix of catalogs.
	//printf("%.15lf %.15lf %.15lf\n", data[0][i], data[1][i], data[2][i]);
    }

    fclose(csv);

    double params[3] = {omega_m, omega_ee, w};

    double chi2 = calc_chi(h0, nrows, data, params, precision);

    return chi2;
}

int main(){
    double h0 = 70;
    double omega_m = 0.3;
    double omega_ee = 0.7;
    double w = -1;
    double precision = 1E-10;
    int nrows = 580;

    char fl_name[MAX] = "fake_data.cat";

    double chi2 = main_execution(fl_name, h0, omega_m, omega_ee, w, precision, nrows);

    printf("%e\n", chi2);

    return 0;
}
