// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
using namespace arma;
using namespace std;

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// simple example of creating two matrices and
// returning the result of an operatioon on them
//
// via the exports attribute we tell Rcpp to make this function
// available from R

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
vec rowsum_Mat(mat M) {
    int nr=M.n_rows;
    vec out(nr);
    for(int i=0;i<nr;i++){
        out(i)=sum(M.row(i));
    }
    return out;
}

// [[Rcpp::export]]
vec colsum_Mat(mat M) {
    int nc=M.n_cols;
    vec out(nc);
    for(int i=0;i<nc;i++){
        out(i)=sum(M.col(i));
    }
    return out;
}

// [[Rcpp::export]]
mat cov_beta_mul(mat cov,mat beta) {
    int cov_nr=cov.n_rows;
    int cov_nc=cov.n_cols;
    int beta_nc=beta.n_cols;
    mat cov_beta(cov_nr,beta_nc,fill::zeros);
    for(int i=0;i<cov_nr;i++){
        for(int j=0;j<beta_nc;j++){
            for(int k=0;k<cov_nc;k++){
                cov_beta(i,j)+=cov(i,k)*beta(k,j);
            }
        }
    }
    return cov_beta;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gamma, Tau update function, gradient and hessian functions and ELBO convergence function

// [[Rcpp::export]]
cube gamma_U_update_stat_undir(mat gamma_U, mat gamma_P, vec pi_U, mat theta, cube logitInv_delta_0, mat net_adjacency, mat net_rating, mat cov_beta_u, mat cov_beta_p, int N1, int N2, int K1, int K2, int R){
    cube quad_lin_coeff(N1,K1,2);
    for(int i = 0; i < N1; i++){
        for(int k = 0; k < K1; k++){
            float t1=0;
            for(int j = 0; j < N2; j++){
                for(int l = 0; l < K2; l++){
                    int indicator_0=(net_adjacency(i,j)==0);
                    float exp_val=exp(theta(k,l)+cov_beta_u(i,k)+cov_beta_p(j,l));
                    if(indicator_0){
                        t1+=(gamma_P(j,l)/(2*gamma_U(i,k)))*(-log(1+exp_val));
                    }else{
                        if(net_rating(i,j)==1){
                            t1+=(gamma_P(j,l)/(2*gamma_U(i,k)))*(log(exp_val/(1+exp_val))+(log(logitInv_delta_0(k,l,0))));
                        }else{
                            t1+=(gamma_P(j,l)/(2*gamma_U(i,k)))*(log(exp_val/(1+exp_val))+(log(logitInv_delta_0(k,l,(net_rating(i,j)-1))-logitInv_delta_0(k,l,(net_rating(i,j)-2)))));
                        }
                    }
                }
            }
            quad_lin_coeff(i,k,0)=t1-(1/gamma_U(i,k));
            quad_lin_coeff(i,k,1)=log(pi_U(k))-log(gamma_U(i,k))+1;
        }
    }
    return quad_lin_coeff;
}

// [[Rcpp::export]]
cube gamma_P_update_stat_undir(mat gamma_U, mat gamma_P, vec pi_P, mat theta, cube logitInv_delta_0, mat net_adjacency, mat net_rating, mat cov_beta_u, mat cov_beta_p, int N1, int N2, int K1, int K2, int R){
    cube quad_lin_coeff(N2,K2,2);
    for(int i = 0; i < N2; i++){
        for(int k = 0; k < K2; k++){
            float t1=0;
            for(int j = 0; j < N1; j++){
                for(int l = 0; l < K1; l++){
                    int indicator_0=(net_adjacency(j,i)==0);
                    float exp_val=exp(theta(l,k)+cov_beta_u(j,l)+cov_beta_p(i,k));
                    if(indicator_0){
                        t1+=(gamma_U(j,l)/(2*gamma_P(i,k)))*(-log(1+exp_val));
                    }else{
                        if(net_rating(j,i)==1){
                            t1+=(gamma_U(j,l)/(2*gamma_P(i,k)))*(log(exp_val/(1+exp_val))+(log(logitInv_delta_0(l,k,0))));
                        }else{
                            t1+=(gamma_U(j,l)/(2*gamma_P(i,k)))*(log(exp_val/(1+exp_val))+(log(logitInv_delta_0(l,k,(net_rating(j,i)-1))-logitInv_delta_0(l,k,(net_rating(j,i)-2)))));
                        }
                    }
				}
            }
            quad_lin_coeff(i,k,0)=t1-(1/gamma_P(i,k));
            quad_lin_coeff(i,k,1)=log(pi_P(k))-log(gamma_P(i,k))+1;
        }
    }
    return quad_lin_coeff;
}

// [[Rcpp::export]]
mat grad_bipartite_stat_undir_theta(mat theta, mat gamma_U, mat gamma_P, mat net_adjacency, mat cov_beta_u, mat cov_beta_p, int N1, int N2, int K1, int K2){
    mat grad_mat(K1,K2,fill::zeros);
    for(int i = 0; i < N1; i++){
        for(int j = 0; j < N2; j++){
            mat grad_matsub(K1,K2);
            for(int k = 0; k < K1; k++){
                for(int l = 0; l < K2; l++){
                    bool indicator_0=(net_adjacency(i,j)==0);
                    float exp_val=exp(theta(k,l)+cov_beta_u(i,k)+cov_beta_p(j,l));
                    grad_matsub(k,l)=gamma_U(i,k)*gamma_P(j,l)*(!indicator_0-(exp_val/(1+exp_val)));
                }
            }
            grad_mat+=grad_matsub;
        }
    }
    return grad_mat;
}

// [[Rcpp::export]]
mat hess_bipartite_stat_undir_theta(mat theta, mat gamma_U, mat gamma_P, mat cov_beta_u, mat cov_beta_p, int N1, int N2, int K1, int K2){
    mat hess_mat(K1,K2,fill::zeros);
    for(int i = 0; i < N1; i++){
        for(int j = 0; j < N2; j++){
            mat hess_matsub(K1,K2);
            for(int k = 0; k < K1; k++){
                for(int l = 0; l < K2; l++){
                    float exp_val=exp(theta(k,l)+cov_beta_u(i,k)+cov_beta_p(j,l));
                    hess_matsub(k,l)=-(gamma_U(i,k)*gamma_P(j,l)*(exp_val/pow((1+exp_val),2)));
                }
            }
            hess_mat+=hess_matsub;
        }
    }
    return hess_mat;
}

// [[Rcpp::export]]
mat grad_bipartite_stat_undir_beta_u(mat gamma_U, mat gamma_P, mat theta, mat net_adjacency, mat cov_beta_u, mat cov_beta_p, mat cov_u_outer, int N1, int N2, int K1, int K2, int C1_outer){
    mat grad_mat(K1,C1_outer,fill::zeros);
    for(int cov_u_index = 0; cov_u_index < C1_outer; cov_u_index++){
        for(int k = 0; k < K1; k++){
            for(int l = 0; l < K2; l++){
                for(int i = 0; i < N1; i++){
                    for(int j = 0; j < N2; j++){
                        bool indicator_0=(net_adjacency(i,j)==0);
                        float exp_val=exp(theta(k,l)+cov_beta_u(i,k)+cov_beta_p(j,l));
                        float grad_cov_common=gamma_U(i,k)*gamma_P(j,l)*(!indicator_0-(exp_val/(1+exp_val)));
                        grad_mat(k,cov_u_index)+=grad_cov_common*cov_u_outer(i,cov_u_index);
                    }
                }
            }
        }
    }
    return grad_mat;
}

// [[Rcpp::export]]
mat hess_bipartite_stat_undir_beta_u(mat gamma_U, mat gamma_P, mat theta, mat cov_beta_u, mat cov_beta_p, mat cov_u_outer, int N1, int N2, int K1, int K2, int C1_outer){
    mat hess_mat(K1,C1_outer,fill::zeros);
    for(int cov_u_index = 0; cov_u_index < C1_outer; cov_u_index++){
        for(int k = 0; k < K1; k++){
            for(int l = 0; l < K2; l++){
                for(int i = 0; i < N1; i++){
                    for(int j = 0; j < N2; j++){
                        float exp_val=exp(theta(k,l)+cov_beta_u(i,k)+cov_beta_p(j,l));
                        float hess_cov_common=-(gamma_U(i,k)*gamma_P(j,l)*(exp_val/(pow((1+exp_val),2))));
                        hess_mat(k,cov_u_index)+=hess_cov_common*pow((cov_u_outer(i,cov_u_index)),2);
                    }
                }
            }
        }
    }
    return hess_mat;
}

// [[Rcpp::export]]
mat grad_bipartite_stat_undir_beta_p(mat gamma_U, mat gamma_P, mat theta, mat net_adjacency, mat cov_beta_u, mat cov_beta_p, mat cov_p_outer, int N1, int N2, int K1, int K2, int C2_outer){
    mat grad_mat(K2,C2_outer,fill::zeros);
    for(int cov_p_index = 0; cov_p_index < C2_outer; cov_p_index++){
        for(int l = 0; l < K2; l++){
            for(int k = 0; k < K1; k++){
                for(int i = 0; i < N1; i++){
                    for(int j = 0; j < N2; j++){
                        bool indicator_0=(net_adjacency(i,j)==0);
                        float exp_val=exp(theta(k,l)+cov_beta_u(i,k)+cov_beta_p(j,l));
                        float grad_cov_common=gamma_U(i,k)*gamma_P(j,l)*(!indicator_0-(exp_val/(1+exp_val)));
                        grad_mat(l,cov_p_index)+=grad_cov_common*cov_p_outer(j,cov_p_index);
                    }
                }
            }
        }
    }
    return grad_mat;
}

// [[Rcpp::export]]
mat hess_bipartite_stat_undir_beta_p(mat gamma_U, mat gamma_P, mat theta, mat cov_beta_u, mat cov_beta_p, mat cov_p_outer, int N1, int N2, int K1, int K2, int C2_outer){
    mat hess_mat(K2,C2_outer,fill::zeros);
    for(int cov_p_index = 0; cov_p_index < C2_outer; cov_p_index++){
        for(int l = 0; l < K2; l++){
            for(int k = 0; k < K1; k++){
                for(int i = 0; i < N1; i++){
                    for(int j = 0; j < N2; j++){
                        float exp_val=exp(theta(k,l)+cov_beta_u(i,k)+cov_beta_p(j,l));
                        float hess_cov_common=-(gamma_U(i,k)*gamma_P(j,l)*(exp_val/(pow((1+exp_val),2))));
                        hess_mat(l,cov_p_index)+=hess_cov_common*pow((cov_p_outer(j,cov_p_index)),2);
                    }
                }
            }
        }
    }
    return hess_mat;
}

// Calculating logit inverse of delta_0
// [[Rcpp::export]]
cube logitInv_delta_0_cal(cube delta_0, int K1, int K2, int R){
    cube logitInv_delta_0(K1,K2,R,fill::zeros);
    for(int k = 0; k < K1; k++){
        for(int l = 0; l < K2; l++){
            for(int r = 0; r < (R-1); r++){
                if(delta_0(k,l,r)>0){
                    logitInv_delta_0(k,l,r)=1/(1+exp(-delta_0(k,l,r)));
                }else{
                    logitInv_delta_0(k,l,r)=(exp(delta_0(k,l,r)))/(1+exp(delta_0(k,l,r)));
                }
            }
            logitInv_delta_0(k,l,(R-1))=1;
        }
    }
    return logitInv_delta_0;
}

// [[Rcpp::export]]
cube grad_bipartite_stat_undir_delta_0(mat gamma_U, mat gamma_P, cube delta_0, cube logitInv_delta_0, mat net_adjacency, mat net_rating, int N1, int N2, int K1, int K2, int R){
    cube grad_delta_0(K1,K2,(R-1),fill::zeros);
    for(int k = 0; k < K1; k++){
        for(int l = 0; l < K2; l++){
            for(int r0 = 0; r0 < (R-1); r0++){
                for(int i = 0; i < N1; i++){
                    for(int j = 0; j < N2; j++){
                        int indicator_0=(net_adjacency(i,j)==0);
                        if(!indicator_0){
                            float f_kl_rij=pow(10,100);
                            if(net_rating(i,j)==1){
                                f_kl_rij=logitInv_delta_0(k,l,0);
                            }else if(net_rating(i,j)==R){
                                f_kl_rij=(1-logitInv_delta_0(k,l,(R-2)));
                            }else{
                                f_kl_rij=(logitInv_delta_0(k,l,(net_rating(i,j)-1))-logitInv_delta_0(k,l,(net_rating(i,j)-2)));
                            }
                            if(r0==(net_rating(i,j)-1)){
                                grad_delta_0(k,l,r0)+=((gamma_U(i,k)*gamma_P(j,l))*(1/f_kl_rij)*(logitInv_delta_0(k,l,r0)/(1+exp(delta_0(k,l,r0)))));
                            }else if((r0+1)==(net_rating(i,j)-1)){
                                grad_delta_0(k,l,r0)+=((gamma_U(i,k)*gamma_P(j,l))*(1/f_kl_rij)*(-logitInv_delta_0(k,l,r0)/(1+exp(delta_0(k,l,r0)))));
                            }
                        }
                    }
                }
            }
        }
    }
    return grad_delta_0;
}

// [[Rcpp::export]]
mat hess_bipartite_stat_undir_delta_0(int k, int l, mat gamma_U, mat gamma_P, cube delta_0, cube logitInv_delta_0, mat net_adjacency, mat net_rating, int N1, int N2, int R){
    mat hess_delta_0((R-1),(R-1),fill::zeros);
    for(int r1 = 0; r1 < (R-1); r1++){
        for(int r2 = r1; r2 < (R-1); r2++){
            for(int i = 0; i < N1; i++){
                for(int j = 0; j < N2; j++){
                    int indicator_0=(net_adjacency(i,j)==0);
                    if(!indicator_0){
                        float f_kl_rij=0;
                        if(net_rating(i,j)==1){
                            f_kl_rij=logitInv_delta_0(k,l,0);
                        }else if(net_rating(i,j)==R){
                            f_kl_rij=(1-logitInv_delta_0(k,l,(R-2)));
                        }else{
                            f_kl_rij=(logitInv_delta_0(k,l,(net_rating(i,j)-1))-logitInv_delta_0(k,l,(net_rating(i,j)-2)));
                        }
                        if(r1==r2){
                            int r0=r1;
                            if(r0==(net_rating(i,j)-1)){
                                float term_1=((1/f_kl_rij)*(logitInv_delta_0(k,l,r0)*((1-exp(delta_0(k,l,r0)))/(pow((1+exp(delta_0(k,l,r0))),2)))));
                                float term_2=pow(((1/f_kl_rij)*(logitInv_delta_0(k,l,r0)/(1+exp(delta_0(k,l,r0))))),2);
                                hess_delta_0(r1,r2)+=((gamma_U(i,k)*gamma_P(j,l))*(term_1-term_2));
                            }else if((r0+1)==(net_rating(i,j)-1)){
                                float term_1=((1/f_kl_rij)*(-logitInv_delta_0(k,l,r0)*((1-exp(delta_0(k,l,r0)))/(pow((1+exp(delta_0(k,l,r0))),2)))));
                                float term_2=pow(((1/f_kl_rij)*(logitInv_delta_0(k,l,r0)/(1+exp(delta_0(k,l,r0))))),2);
                                hess_delta_0(r1,r2)+=((gamma_U(i,k)*gamma_P(j,l))*(term_1-term_2));
                            }
                        }else{
                            float numerator_r1=0;
                            if(r1==(net_rating(i,j)-1)){
                                numerator_r1=(logitInv_delta_0(k,l,r1)/(1+exp(delta_0(k,l,r1))));
                            }else if((r1+1)==(net_rating(i,j)-1)){
                                numerator_r1=(-logitInv_delta_0(k,l,r1)/(1+exp(delta_0(k,l,r1))));
                            }
                            
                            float numerator_r2=0;
                            if(r2==(net_rating(i,j)-1)){
                                numerator_r2=(logitInv_delta_0(k,l,r2)/(1+exp(delta_0(k,l,r2))));
                            }else if((r2+1)==(net_rating(i,j)-1)){
                                numerator_r2=(-logitInv_delta_0(k,l,r2)/(1+exp(delta_0(k,l,r2))));
                            }
                            
                            hess_delta_0(r1,r2)+=((gamma_U(i,k)*gamma_P(j,l))*(-(numerator_r1*numerator_r2)/(pow(f_kl_rij,2))));
                        }
                    }
                }
            }
            hess_delta_0(r2,r1)=hess_delta_0(r1,r2);
        }
    }
    return hess_delta_0;
}


// [[Rcpp::export]]
float ELBO_conv_bipartite_stat_undir(mat gamma_U, mat gamma_P, vec pi_U, vec pi_P, mat theta, cube logitInv_delta_0, mat net_adjacency, mat net_rating, mat cov_beta_u, mat cov_beta_p, int N1, int N2, int K1, int K2){
    float t1=0;
    for(int i = 0; i < N1; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < K1; k++){
                for(int l = 0; l < K2; l++){
                    int indicator_0=(net_adjacency(i,j)==0);
                    float exp_val=exp(theta(k,l)+cov_beta_u(i,k)+cov_beta_p(j,l));
                    if(indicator_0){
                       t1+=(gamma_U(i,k)*gamma_P(j,l)*(-log(1+exp_val)));
                    }else{
                        if(net_rating(i,j)==1){
                            t1+=(gamma_U(i,k)*gamma_P(j,l)*(log(exp_val/(1+exp_val))+(log(logitInv_delta_0(k,l,0)))));
                        }else{
                            t1+=(gamma_U(i,k)*gamma_P(j,l)*(log(exp_val/(1+exp_val))+(log(logitInv_delta_0(k,l,(net_rating(i,j)-1))-logitInv_delta_0(k,l,(net_rating(i,j)-2))))));
                        }
                        
                    }
				}
            }
        }
    }
    float t2=0;
    for(int i = 0; i < N1; i++){
        for(int k = 0; k < K1; k++){
            if((pi_U(k)>=(pow(10,(-100))))&(gamma_U(i,k)>=(pow(10,(-100))))){
                t2+=gamma_U(i,k)*(log(pi_U(k))-log(gamma_U(i,k)));
            }
        }
    }
    float t3=0;
    for(int j = 0; j < N2; j++){
        for(int l = 0; l < K2; l++){
            if((pi_P(l)>=(pow(10,(-100))))&(gamma_P(j,l)>=(pow(10,(-100))))){
                t3+=gamma_P(j,l)*(log(pi_P(l))-log(gamma_P(j,l)));
            }
        }
    }
    float ELBO_val=t1+t2+t3;
    return ELBO_val;
}
///////////////////////////////////////////////////////////////////////////////////////////////
