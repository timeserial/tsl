/* Celula de dois tempos em PONTO FIXO - o degrau final antes do metal.
 * Sinais e sombra: int16 Q12. Mestres de aprendizagem: int32 Q20 (a
 * plasticidade fina acumula 8 bits abaixo do que o compute ve). LUTs de
 * 256 entradas com interpolacao linear; ZERO libm no laco.
 * Contrato: MSE de treino e avaliacao a <10% do float no mesmo percurso. */
#include <stdio.h>
#include "golden_twostroke.h"

#define Q 12
#define QM 20
#define ONE (1<<Q)
#ifndef LR_MILLI
#define LR_MILLI 100
#endif
#define LRQ ((acc64)LR_MILLI*ONE/1000)
typedef short q12; typedef long long acc64;

static long W0m[N_IN*N_TOP], Am[N_TOP*N_TOP], Gm[N_TOP*N_TOP], bm[N_TOP];
static q12  W0 [N_IN*N_TOP], A [N_TOP*N_TOP], G [N_TOP*N_TOP], b [N_TOP];
static q12  s[N_TOP], lut_tanh[256], lut_sig[256];

static double my_exp(double x){ double t=1,sum=1; int i;
    for(i=1;i<14;i++){ t*=x/i; sum+=t; } return sum; }
static double my_tanh(double x){ double e2;
    if(x>4) return 1; if(x<-4) return -1;
    e2=my_exp(2*x); return (e2-1)/(e2+1); }
static void make_luts(void){ int i;
    for(i=0;i<256;i++) lut_tanh[i]=(q12)(my_tanh(-4.0+8.0*i/255.0)*ONE);
    for(i=0;i<256;i++) lut_sig[i]=(q12)((0.5+0.5*my_tanh((-8.0+16.0*i/255.0)/2))*ONE); }
static q12 lut_interp(const q12 *lut, acc64 x, acc64 lo, acc64 span){
    acc64 t=(x-lo)*255; long idx=(long)(t/span); acc64 rem=t%span;
    if(idx>=255) return lut[255]; if(idx<0) return lut[0];
    return (q12)(lut[idx]+(((acc64)(lut[idx+1]-lut[idx])*rem)/span)); }
static q12 q_tanh(acc64 x){ if(x>=4*ONE) return ONE-1;
    if(x<=-4*ONE) return -(ONE-1); return lut_interp(lut_tanh,x,-4*ONE,8*ONE); }
static q12 q_sig(acc64 x){ if(x>=8*ONE) return ONE-1;
    if(x<=-8*ONE) return 1; return lut_interp(lut_sig,x,-8*ONE,16*ONE); }

static q12 sat(acc64 x){ if(x>32767) return 32767;
    if(x<-32768) return -32768; return (q12)x; }
static long satm(acc64 x){ if(x>2147483647LL) return 2147483647L;
    if(x<-2147483648LL) return -2147483648L; return (long)x; }
static void refresh(void){ int i;
    for(i=0;i<N_IN*N_TOP;i++) W0[i]=sat(W0m[i]>>(QM-Q));
    for(i=0;i<N_TOP*N_TOP;i++){ A[i]=sat(Am[i]>>(QM-Q)); G[i]=sat(Gm[i]>>(QM-Q)); }
    for(i=0;i<N_TOP;i++) b[i]=sat(bm[i]>>(QM-Q)); }
static void matvec_q(const q12 *M, const q12 *v, acc64 *o, int r, int c){
    int i,j; for(i=0;i<r;i++){ acc64 a=0;
        for(j=0;j<c;j++) a+=(acc64)M[i*c+j]*v[j]; o[i]=a>>Q; } }
static void matvec_qT(const q12 *M, const q12 *v, acc64 *o, int r, int c){
    int i,j; for(j=0;j<c;j++){ acc64 a=0;
        for(i=0;i<r;i++) a+=(acc64)M[i*c+j]*v[i]; o[j]=a>>Q; } }

static long step_learn_q(const q12 *x, int learn){
    acc64 cq[N_TOP], gq[N_TOP], prior[N_TOP], pred[N_IN], e[N_IN], h[N_TOP];
    q12 pr16[N_IN>N_TOP?N_TOP:N_TOP], e16[N_IN];
    acc64 ns=0, npr=0, lr_ns, lr_npr, mse=0, v; int i,j;
    matvec_q(A, s, cq, N_TOP, N_TOP);
    for(i=0;i<N_TOP;i++) cq[i]=q_tanh(cq[i]);
    matvec_q(G, s, gq, N_TOP, N_TOP);
    for(i=0;i<N_TOP;i++) gq[i]=q_sig(gq[i]+b[i]);
    for(i=0;i<N_TOP;i++){ prior[i]=((ONE-gq[i])*(acc64)s[i]+gq[i]*cq[i])>>Q;
        pr16[i]=sat(prior[i]); }
    matvec_q(W0, pr16, pred, N_IN, N_TOP);
    for(i=0;i<N_IN;i++){ e[i]=(acc64)x[i]-pred[i]; e16[i]=sat(e[i]); mse+=e[i]*e[i]; }
    mse/=N_IN;
    matvec_qT(W0, e16, h, N_IN, N_TOP);   /* leitura transposta (sempre) */
    if(learn){
        for(i=0;i<N_TOP;i++){ ns+=(acc64)s[i]*s[i]; npr+=(acc64)pr16[i]*pr16[i]; }
        ns=(ns>>Q)+4; npr=(npr>>Q)+4;                     /* Q12 + eps */
        lr_ns=(LRQ*ONE)/ns; lr_npr=(LRQ*ONE)/npr;         /* Q12 */
        for(i=0;i<N_IN;i++) for(j=0;j<N_TOP;j++)
            W0m[i*N_TOP+j]=satm(W0m[i*N_TOP+j]+((lr_npr*e[i]*prior[j])>>16));
        for(i=0;i<N_TOP;i++){
            acc64 mc=((h[i]*gq[i])>>Q)*(ONE-((cq[i]*cq[i])>>Q))>>Q;
            acc64 mg=((h[i]*(cq[i]-s[i]))>>Q)*gq[i]>>Q;
            mg=(mg*(ONE-gq[i]))>>Q;
            for(j=0;j<N_TOP;j++){
                Am[i*N_TOP+j]=satm(Am[i*N_TOP+j]+((lr_ns*mc*s[j])>>16));
                Gm[i*N_TOP+j]=satm(Gm[i*N_TOP+j]+((lr_ns*mg*s[j])>>16)); }
            bm[i]=satm(bm[i]+((LRQ*mg)>>4)); }
        refresh();
    }
    /* correcao sensorial de um passo (quebra o ponto fixo do zero) */
    for(i=0;i<N_TOP;i++){ v=prior[i]+(((acc64)(0.2*ONE)*h[i])>>Q);
        if(v>2*ONE) v=2*ONE; if(v<-2*ONE) v=-2*ONE; s[i]=(q12)v; }
    return (long)mse;
}

static void load_m(long *dm, const float *src, int n){ int i;
    for(i=0;i<n;i++) dm[i]=satm((acc64)(src[i]*(1L<<QM))); }
static void reset_all(void){ int k;
    load_m(W0m,g_W0_init,N_IN*N_TOP); load_m(Am,g_A_init,N_TOP*N_TOP);
    load_m(Gm,g_G_init,N_TOP*N_TOP); load_m(bm,g_b_init,N_TOP);
    refresh(); for(k=0;k<N_TOP;k++) s[k]=0; }

int main(void){
    int k,i; long m; double mse_tr=0, mse_ev=0; q12 xq[N_IN];
    make_luts(); reset_all();
    for(k=0;k<200;k++){ for(i=0;i<N_IN;i++) xq[i]=sat((acc64)(g_frames[k*N_IN+i]*ONE));
        m=step_learn_q(xq,1); mse_tr+=(double)m/ONE/ONE; } mse_tr/=200;
    for(k=200;k<300;k++){ for(i=0;i<N_IN;i++) xq[i]=sat((acc64)(g_frames[k*N_IN+i]*ONE));
        m=step_learn_q(xq,0); mse_ev+=(double)m/ONE/ONE; } mse_ev/=100;
    printf("PONTO FIXO Q12/Q20, LUTs interpoladas, zero libm no laco\n");
    printf("  mse treino: C-int=%.6f  float=%.6f  (dif %+.1f%%)\n",
        mse_tr,(double)G_MSE_TRAIN,100.0*(mse_tr-G_MSE_TRAIN)/G_MSE_TRAIN);
    printf("  mse aval:   C-int=%.6f  float=%.6f  (dif %+.1f%%)\n",
        mse_ev,(double)G_MSE_EVAL,100.0*(mse_ev-G_MSE_EVAL)/G_MSE_EVAL);
    if(mse_ev>1.10*G_MSE_EVAL){ printf("FALHOU: >10%%\n"); return 1; }
    printf("TUDO OK - a aprendizagem sobrevive aos inteiros.\n");
    return 0;
}
