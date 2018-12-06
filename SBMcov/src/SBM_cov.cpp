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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gamma, Tau update function, gradient and hessian functions and ELBO convergence function

// [[Rcpp::export]]
cube gamma_update_stat_undir(mat gamma, vec pi, mat theta, cube logitInv_delta_0, mat net_adjacency, mat net_rating, int N, int K, int R){
    cube quad_lin_coeff(N,K,2);
    for(int i = 0; i < N; i++){
        if(i!=(N-1)){
            for(int k = 0; k < K; k++){
                float t1=0;
                for(int j = (i+1); j < N; j++){
                    for(int l = 0; l < K; l++){
                        int indicator_0=(net_adjacency(i,j)==0);
                        float exp_val=exp(theta(k,l));
                        if(indicator_0){
                            t1+=(gamma(j,l)/(2*gamma(i,k)))*(-log(1+exp_val));
                        }else{
                            if(net_rating(i,j)==1){
                                t1+=(gamma(j,l)/(2*gamma(i,k)))*(log(exp_val/(1+exp_val))+(log(logitInv_delta_0(k,l,0))));
                            }else{
                                t1+=(gamma(j,l)/(2*gamma(i,k)))*(log(exp_val/(1+exp_val))+(log(logitInv_delta_0(k,l,(net_rating(i,j)-1))-logitInv_delta_0(k,l,(net_rating(i,j)-2)))));
                            }
                        }
                    }
                }
                quad_lin_coeff(i,k,0)=t1-(1/gamma(i,k));
                quad_lin_coeff(i,k,1)=log(pi(k))-log(gamma(i,k))+1;
            }
        }else if(i==(N-1)){
            for(int k = 0; k < K; k++){
                quad_lin_coeff(i,k,0)=-(1/gamma((N-1),k));
                quad_lin_coeff(i,k,1)=log(pi(k))-log(gamma((N-1),k))+1;
            }
        }
    }
    return quad_lin_coeff;
}

// [[Rcpp::export]]
mat grad_bipartite_stat_undir_theta(mat theta, mat gamma, mat net_adjacency, int N, int K){
    mat grad_mat(K,K,fill::zeros);
    for(int i = 0; i < (N-1); i++){
        for(int j = (i+1); j < N; j++){
            mat grad_matsub(K,K);
            for(int k = 0; k < K; k++){
                for(int l = 0; l < K; l++){
                    bool indicator_0=(net_adjacency(i,j)==0);
                    float exp_val=exp(theta(k,l));
                    grad_matsub(k,l)=gamma(i,k)*gamma(j,l)*(!indicator_0-(exp_val/(1+exp_val)));
                }
            }
            grad_mat+=grad_matsub;
        }
    }
    return grad_mat;
}

// [[Rcpp::export]]
mat hess_bipartite_stat_undir_theta(mat theta, mat gamma, int N, int K){
    mat hess_mat(K,K,fill::zeros);
    for(int i = 0; i < (N-1); i++){
        for(int j = (i+1); j < N; j++){
            mat hess_matsub(K,K);
            for(int k = 0; k < K; k++){
                for(int l = 0; l < K; l++){
                    float exp_val=exp(theta(k,l));
                    hess_matsub(k,l)=-(gamma(i,k)*gamma(j,l)*(exp_val/pow((1+exp_val),2)));
                }
            }
            hess_mat+=hess_matsub;
        }
    }
    return hess_mat;
}

// Calculating logit inverse of delta_0
// [[Rcpp::export]]
cube logitInv_delta_0_cal(cube delta_0, int K, int R){
    cube logitInv_delta_0(K,K,R,fill::zeros);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
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
cube grad_bipartite_stat_undir_delta_0(mat gamma, cube delta_0, cube logitInv_delta_0, mat net_adjacency, mat net_rating, int N, int K, int R){
    cube grad_delta_0(K,K,(R-1),fill::zeros);
    for(int k = 0; k < K; k++){
        for(int l = 0; l < K; l++){
            for(int r0 = 0; r0 < (R-1); r0++){
                for(int i = 0; i < (N-1); i++){
                    for(int j = (i+1); j < N; j++){
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
                                if(f_kl_rij==0){
                                    grad_delta_0(k,l,r0)+=-pow(10,100);
                                }else{
                                    grad_delta_0(k,l,r0)+=((gamma(i,k)*gamma(j,l))*(1/f_kl_rij)*(logitInv_delta_0(k,l,r0)/(1+exp(delta_0(k,l,r0)))));
                                }
                            }else if((r0+1)==(net_rating(i,j)-1)){
                                if(f_kl_rij==0){
                                    grad_delta_0(k,l,r0)+=-pow(10,100);
                                }else{
                                    grad_delta_0(k,l,r0)+=((gamma(i,k)*gamma(j,l))*(1/f_kl_rij)*(-logitInv_delta_0(k,l,r0)/(1+exp(delta_0(k,l,r0)))));
                                }
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
mat hess_bipartite_stat_undir_delta_0(int k, int l, mat gamma, cube delta_0, cube logitInv_delta_0, mat net_adjacency, mat net_rating, int N, int R){
    mat hess_delta_0((R-1),(R-1),fill::zeros);
    for(int r1 = 0; r1 < (R-1); r1++){
        for(int r2 = r1; r2 < (R-1); r2++){
            for(int i = 0; i < (N-1); i++){
                for(int j = (i+1); j < N; j++){
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
                        //cout<<f_kl_rij<<endl;
                        if(r1==r2){
                            int r0=r1;
                            if(r0==(net_rating(i,j)-1)){
                                if(f_kl_rij==0){
                                    hess_delta_0(r1,r2)+=pow(10,100);
                                }else{
                                    float term_1=((1/f_kl_rij)*(logitInv_delta_0(k,l,r0)*((1-exp(delta_0(k,l,r0)))/(pow((1+exp(delta_0(k,l,r0))),2)))));
                                    float term_2=pow(((1/f_kl_rij)*(logitInv_delta_0(k,l,r0)/(1+exp(delta_0(k,l,r0))))),2);
                                    hess_delta_0(r1,r2)+=((gamma(i,k)*gamma(j,l))*(term_1-term_2));
                                }
                            }else if((r0+1)==(net_rating(i,j)-1)){
                                if(f_kl_rij==0){
                                    hess_delta_0(r1,r2)+=pow(10,100);
                                }else{
                                    float term_1=((1/f_kl_rij)*(-logitInv_delta_0(k,l,r0)*((1-exp(delta_0(k,l,r0)))/(pow((1+exp(delta_0(k,l,r0))),2)))));
                                    float term_2=pow(((1/f_kl_rij)*(logitInv_delta_0(k,l,r0)/(1+exp(delta_0(k,l,r0))))),2);
                                    hess_delta_0(r1,r2)+=((gamma(i,k)*gamma(j,l))*(term_1-term_2));
                                }
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
                            if(f_kl_rij==0){
                                hess_delta_0(r1,r2)+=pow(10,100);
                            }else{
                                hess_delta_0(r1,r2)+=((gamma(i,k)*gamma(j,l))*(-(numerator_r1*numerator_r2)/(pow(f_kl_rij,2))));
                            }
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
float ELBO_conv_bipartite_stat_undir(mat gamma, vec pi, mat theta, cube logitInv_delta_0, mat net_adjacency, mat net_rating, int N, int K){
    float t1=0;
    for(int i = 0; i < (N-1); i++){
        for(int j = (i+1); j < N; j++){
            for(int k = 0; k < K; k++){
                for(int l = 0; l < K; l++){
                    int indicator_0=(net_adjacency(i,j)==0);
                    float exp_val=exp(theta(k,l));
                    if(indicator_0){
                       t1+=(gamma(i,k)*gamma(j,l)*(-log(1+exp_val)));
                    }else{
                        if(net_rating(i,j)==1){
                            t1+=(gamma(i,k)*gamma(j,l)*(log(exp_val/(1+exp_val))+(log(logitInv_delta_0(k,l,0)))));
                        }else{
                            t1+=(gamma(i,k)*gamma(j,l)*(log(exp_val/(1+exp_val))+(log(logitInv_delta_0(k,l,(net_rating(i,j)-1))-logitInv_delta_0(k,l,(net_rating(i,j)-2))))));
                        }
                        
                    }
				}
            }
        }
    }
    float t2=0;
    for(int i = 0; i < N; i++){
        for(int k = 0; k < K; k++){
            if((pi(k)>=(pow(10,(-100))))&(gamma(i,k)>=(pow(10,(-100))))){
                t2+=gamma(i,k)*(log(pi(k))-log(gamma(i,k)));
            }
        }
    }
    float ELBO_val=t1+t2;
    return ELBO_val;
}
///////////////////////////////////////////////////////////////////////////////////////////////
