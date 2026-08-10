/* A célula de dois tempos em C99 puro.
 *
 * O teste honesto do plano: se não couber em C simples — arrays estáticos,
 * zero malloc, zero dependências — não cabe no crossbar. Cabe:
 *   - previsão:       1 matvec + 2 matvecs pequenos + 2 LUTs
 *   - crédito local:  1 matvec transposto + 3 produtos exteriores
 *   - memória total:  ~11 KB em float32 (N_IN=64, N_TOP=24)
 *
 * Compila igual no Mac e no ESP32:  cc -std=c99 -O2 twostroke.c -lm
 * Valida contra vetores dourados do Python (contrato em c/README.md):
 *   1) uma atualização isolada: exata a 1e-5;
 *   2) trajetória de 300 janelas: MSE agregado dentro de 2%.
 */
#include <math.h>
#include <stdio.h>
#include "golden_twostroke.h"

static float W0[N_IN*N_TOP], A[N_TOP*N_TOP], G[N_TOP*N_TOP], b[N_TOP];
static float s[N_TOP];

/* ---- primitivas: as três operações do substrato ---------------------- */
static void matvec(const float *M, const float *v, float *out, int r, int c){
    int i,j; for(i=0;i<r;i++){ float a=0; for(j=0;j<c;j++) a+=M[i*c+j]*v[j]; out[i]=a; }
}
static void matvec_T(const float *M, const float *v, float *out, int r, int c){
    int i,j; for(j=0;j<c;j++){ float a=0; for(i=0;i<r;i++) a+=M[i*c+j]*v[i]; out[j]=a; }
}
static void rank1(float *M, const float *u, const float *v, float lr, int r, int c){
    int i,j; for(i=0;i<r;i++) for(j=0;j<c;j++) M[i*c+j]+=lr*u[i]*v[j];
}
static float dot(const float *a, const float *bb, int n){
    int i; float x=0; for(i=0;i<n;i++) x+=a[i]*bb[i]; return x;
}

/* ---- um passo: prever, creditar (fase 1), transitar ------------------- */
static float step_learn(const float *x, float lr){
    float c_[N_TOP], g_[N_TOP], prior[N_TOP], pred[N_IN], e[N_IN], h[N_TOP];
    float mc[N_TOP], mg[N_TOP];
    float ns, npr, mse=0; int i;
    matvec(A, s, c_, N_TOP, N_TOP);
    for(i=0;i<N_TOP;i++) c_[i]=tanhf(c_[i]);
    matvec(G, s, g_, N_TOP, N_TOP);
    for(i=0;i<N_TOP;i++) g_[i]=1.0f/(1.0f+expf(-(g_[i]+b[i])));
    for(i=0;i<N_TOP;i++) prior[i]=(1.0f-g_[i])*s[i]+g_[i]*c_[i];
    matvec(W0, prior, pred, N_IN, N_TOP);
    for(i=0;i<N_IN;i++){ e[i]=x[i]-pred[i]; mse+=e[i]*e[i]; }
    mse/=N_IN;
    if(lr>0){
        matvec_T(W0, e, h, N_IN, N_TOP);            /* leitura transposta   */
        ns=dot(s,s,N_TOP)+1e-6f; npr=dot(prior,prior,N_TOP)+1e-6f;
        rank1(W0, e, prior, lr/npr, N_IN, N_TOP);   /* escritas rank-1      */
        for(i=0;i<N_TOP;i++) mc[i]=h[i]*g_[i]*(1.0f-c_[i]*c_[i]);
        rank1(A, mc, s, lr/ns, N_TOP, N_TOP);
        for(i=0;i<N_TOP;i++) mg[i]=h[i]*(c_[i]-s[i])*g_[i]*(1.0f-g_[i]);
        rank1(G, mg, s, lr/ns, N_TOP, N_TOP);
        for(i=0;i<N_TOP;i++) b[i]+=lr*mg[i];
    }
    for(i=0;i<N_TOP;i++) s[i]=prior[i];            /* transitar             */
    return mse;
}

static void load(float *d, const float *src, int n){ int i; for(i=0;i<n;i++) d[i]=src[i]; }
static float maxdiff(const float *a, const float *bb, int n){
    int i; float m=0, d; for(i=0;i<n;i++){ d=fabsf(a[i]-bb[i]); if(d>m) m=d; } return m;
}

int main(void){
    int k; float m, mse_tr=0, mse_ev=0;
    /* --- contrato 1: uma atualização isolada, exata --------------------- */
    load(W0,g_W0_init,N_IN*N_TOP); load(A,g_A_init,N_TOP*N_TOP);
    load(G,g_G_init,N_TOP*N_TOP);  load(b,g_b_init,N_TOP);
    load(s,g_s_fix,N_TOP);
    step_learn(g_x_fix, LR);
    m=maxdiff(W0,g_W0_after1,N_IN*N_TOP);
    m=fmaxf(m,maxdiff(A,g_A_after1,N_TOP*N_TOP));
    m=fmaxf(m,maxdiff(G,g_G_after1,N_TOP*N_TOP));
    m=fmaxf(m,maxdiff(b,g_b_after1,N_TOP));
    printf("contrato 1 (exatidao 1 passo): maxdiff=%.3g  %s\n",
           (double)m, m<1e-5f?"OK":"FALHOU");
    if(m>=1e-5f) return 1;
    /* --- contrato 2: trajetória agregada -------------------------------- */
    load(W0,g_W0_init,N_IN*N_TOP); load(A,g_A_init,N_TOP*N_TOP);
    load(G,g_G_init,N_TOP*N_TOP);  load(b,g_b_init,N_TOP);
    for(k=0;k<N_TOP;k++) s[k]=0;
    for(k=0;k<200;k++) mse_tr+=step_learn(&g_frames[k*N_IN], LR);
    mse_tr/=200;
    for(k=200;k<300;k++) mse_ev+=step_learn(&g_frames[k*N_IN], 0.0f);
    mse_ev/=100;
    printf("contrato 2: mse_treino C=%.6f py=%.6f  (dif %.2f%%)\n",
           (double)mse_tr,(double)G_MSE_TRAIN,
           100.0*fabs(mse_tr-G_MSE_TRAIN)/G_MSE_TRAIN);
    printf("            mse_aval   C=%.6f py=%.6f  (dif %.2f%%)\n",
           (double)mse_ev,(double)G_MSE_EVAL,
           100.0*fabs(mse_ev-G_MSE_EVAL)/G_MSE_EVAL);
    if(fabs(mse_tr-G_MSE_TRAIN)/G_MSE_TRAIN>0.02) return 2;
    if(fabs(mse_ev-G_MSE_EVAL)/G_MSE_EVAL>0.02) return 3;
    printf("TUDO OK — a celula de dois tempos cabe em C simples.\n");
    return 0;
}
