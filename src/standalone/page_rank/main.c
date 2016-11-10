/* -*- mode: C; mode: folding; fill-column: 70; -*- */
#define _XOPEN_SOURCE 600
#define _LARGEFILE64_SOURCE 1
#define _FILE_OFFSET_BITS 64

#if defined(_OPENMP)
//#include "omp.h"
#endif

#include "stinger_core/stinger_atomics.h"
#include "stinger_utils/stinger_utils.h"

#include "stinger_core/stinger.h"
#include "stinger_core/xmalloc.h"
#include "stinger_core/stinger_error.h"
#include "stinger_net/stinger_alg.h"
#include "stinger_utils/timer.h"

#include "stinger_core/x86_full_empty.h"

#define EPSILON_DEFAULT 1e-8
#define DAMPINGFACTOR_DEFAULT 0.85
#define MAXITER_DEFAULT 20 //50 //20

#define NO_ITERATION_LIMIT 0
#define COUNT_NUM_V_DELTA  0
#define ADD_GRAPH_UPDATE   0
#define SAVE_PR_VALUES     0
#define TEST_EPSLION_THRESHOLD 1

int64_t page_rank_subset(stinger_t * S, int64_t NV, uint8_t * vertex_set, int64_t vertex_set_size, double * pr, double * tmp_pr_in, double epsilon, double dampingfactor, int64_t maxiter);
#if TEST_EPSLION_THRESHOLD == 1
int64_t page_rank_directed(stinger_t * S, int64_t NV, double * pr, double * tmp_pr_in, double epsilon, double dampingfactor, int64_t maxiter, double threshold);
#else
int64_t page_rank_directed(stinger_t * S, int64_t NV, double * pr, double * tmp_pr_in, double epsilon, double dampingfactor, int64_t maxiter);
#endif
int64_t page_rank (stinger_t * S, int64_t NV, double * pr, double * tmp_pr_in, double epsilon, double dampingfactor, int64_t maxiter);
int64_t page_rank_type(stinger_t * S, int64_t NV, double * pr, double * tmp_pr_in, double epsilon, double dampingfactor, int64_t maxiter, int64_t type);
int64_t page_rank_type_directed(stinger_t * S, int64_t NV, double * pr, double * tmp_pr_in, double epsilon, double dampingfactor, int64_t maxiter, int64_t type);


double * set_tmp_pr(double * tmp_pr_in, int64_t NV);
void unset_tmp_pr(double * tmp_pr, double * tmp_pr_in);

#define ACTI(k) (action[2*(k)])
#define ACTJ(k) (action[2*(k)+1])

static int64_t nv, ne, naction;
static int64_t * restrict off;
static int64_t * restrict from;
static int64_t * restrict ind;
static int64_t * restrict weight;
static int64_t * restrict action;

/* handles for I/O memory */
static int64_t * restrict graphmem;
static int64_t * restrict actionmem;

static char * initial_graph_name = "a.18.8.1000.bin";//INITIAL_GRAPH_NAME_DEFAULT;
static char * action_stream_name = "a.18.8.1000.bin";//ACTION_STREAM_NAME_DEFAULT;

static long batch_size = BATCH_SIZE_DEFAULT;
static long nbatch = NBATCH_DEFAULT;

static double threshold;
static double epsilonchange;

static struct stinger * S;
static double * update_time_trace;

FILE* fp; //count
FILE* fp_pr; //pr values
int   fp_idx = 0;


int
main (const int argc, char *argv[])
{
#if 0
  parse_args (argc, argv, &initial_graph_name, &action_stream_name, &batch_size, &nbatch);
  STATS_INIT();

  load_graph_and_action_stream (initial_graph_name, &nv, &ne, (int64_t**)&off,
	      (int64_t**)&ind, (int64_t**)&weight, (int64_t**)&graphmem,
	      action_stream_name, &naction, (int64_t**)&action, (int64_t**)&actionmem);

  print_initial_graph_stats (nv, ne, batch_size, nbatch, naction);
  BATCH_SIZE_CHECK();

#if defined(_OPENMP)
  OMP("omp parallel")
  {
  OMP("omp master")
  PRINT_STAT_INT64 ("num_threads", (long int) omp_get_num_threads());
  }
#endif


  update_time_trace = xmalloc (nbatch * sizeof(*update_time_trace));

  /* Convert to STINGER */
  tic ();
  S = stinger_new ();
  stinger_set_initial_edges (S, nv, 0, off, ind, weight, NULL, NULL, -2);
  PRINT_STAT_DOUBLE ("time_stinger", toc ());
  fflush(stdout);

  free(graphmem);

  tic ();
  uint32_t errorCode = stinger_consistency_check (S, nv);
  double time_check = toc ();
  PRINT_STAT_HEX64 ("error_code", (long unsigned) errorCode);
  PRINT_STAT_DOUBLE ("time_check", timpage_rank_directede_check);

  /* Updates */
  int64_t ntrace = 0;

  for (int64_t actno = 0; actno < nbatch * batch_size; actno += batch_size)
  {
    tic();

    const int64_t endact = (actno + batch_size > naction ? naction : actno + batch_size);
    int64_t *actions = &action[2*actno];
    int64_t numActions = endact - actno;

    MTA("mta assert parallel")
    MTA("mta block dynamic schedule")
    OMP("omp parallel for")
    for(uint64_t k = 0; k < endact - actno; k++) {
      const int64_t i = actions[2 * k];
      const int64_t j = actions[2 * k + 1];

      if (i != j && i < 0) {
	stinger_remove_edge(S, 0, ~i, ~j);
	stinger_remove_edge(S, 0, ~j, ~i);
      }

      if (i != j && i >= 0) {
	stinger_insert_edge (S, 0, i, j, 1, actno+2);
	stinger_insert_edge (S, 0, j, i, 1, actno+2);
      }
    }

    update_time_trace[ntrace] = toc();
    ntrace++;

  } /* End of batch */

  /* Print the times */
  double time_updates = 0;
  for (int64_t k = 0; k < nbatch; k++) {
    time_updates += update_time_trace[k];
  }
  PRINT_STAT_DOUBLE ("time_updates", time_updates);
  PRINT_STAT_DOUBLE ("updates_per_sec", (nbatch * batch_size) / time_updates); 

  tic ();
  errorCode = stinger_consistency_check (S, nv);
  time_check = toc ();
  PRINT_STAT_HEX64 ("error_code", (long unsigned) errorCode);
  PRINT_STAT_DOUBLE ("time_check", time_check);

  free(update_time_trace);
  stinger_free_all (S);
  free (actionmem);
  STATS_END();
#else

#if 0
  int opt = 0;
  while(-1 != (opt = getopt(argc, argv, "t:e:f:i:d?h"))) {
    switch(opt) {
      case 't': {
        sprintf(name, "pagerank_%s", optarg);
        strcpy(type_str,optarg);
        type_specified = 1;
      } break;
      case 'd': {
        directed = 1;
      } break;
      case 'e': {
        epsilon = atof(optarg);
      } break;
      case 'f': {
        dampingfactor = atof(optarg);
      } break;
      case 'i': {
        maxiter = atol(optarg);
      } break;
      default:
        printf("Unknown option '%c'\n", opt);
      case '?':
      case 'h': {
        printf(
          "PageRank\n"
          "==================================\n"
          "\n"
          "  -t <str>  Specify an edge type to run page rank over\n"
          "  -d        Use a PageRank that is safe on directed graphs\n"
          "  -e        Set PageRank Epsilon (default: %0.1e)\n"
          "  -f        Set PageRank Damping Factor (default: %lf)\n"
          "  -i        Set PageRank Max Iterations (default: %ld)\n"
          "\n",EPSILON_DEFAULT,DAMPINGFACTOR_DEFAULT,MAXITER_DEFAULT);
        return(opt);
      }
    }
  }
#endif

  double epsilon = EPSILON_DEFAULT;
  double dampingfactor = DAMPINGFACTOR_DEFAULT;
  int64_t maxiter = MAXITER_DEFAULT;

#if TEST_EPSLION_THRESHOLD == 1
  parse_args_1 (argc, argv, &initial_graph_name, &action_stream_name, &batch_size, &nbatch, &threshold, &epsilonchange);
  STATS_INIT();

  LOG_I_A("!!threshold: %5.10e, epsilon_change: %5.10e", threshold, epsilonchange);
#else
  parse_args (argc, argv, &initial_graph_name, &action_stream_name, &batch_size, &nbatch);
#endif

  STATS_INIT();

  load_graph_and_action_stream (initial_graph_name, &nv, &ne, (int64_t**)&off,
	      (int64_t**)&ind, (int64_t**)&weight, (int64_t**)&graphmem,
	      action_stream_name, &naction, (int64_t**)&action, (int64_t**)&actionmem);

  print_initial_graph_stats (nv, ne, batch_size, nbatch, naction);
  BATCH_SIZE_CHECK();

  double *pr = (double*)xmalloc(sizeof(double) * nv);
  double *tmp_pr = (double*)xmalloc(sizeof(double) * nv);

  tic ();
  S = stinger_new ();//added
  stinger_set_initial_edges (S, nv, 0, off, ind, weight, NULL, NULL, -2);//added
  PRINT_STAT_DOUBLE ("time_stinger", toc ());
#if 1
#if COUNT_NUM_V_DELTA == 1
  fp = fopen("delta_count.txt", "a");
#endif

#if TEST_EPSLION_THRESHOLD == 1
  LOG_I_A("threshold: %5.5e, epsilon: %5.5e", threshold, epsilonchange);
  epsilonchange = pow(10, -(epsilonchange));
  threshold = threshold * 0.000005;// 0.0005;
  LOG_I_A("threshold: %5.5e, epsilon: %5.5e", threshold, epsilonchange);
  page_rank_directed(S, nv, pr, tmp_pr, epsilonchange, dampingfactor, maxiter, threshold);
#else
  page_rank_directed(S, nv, pr, tmp_pr, epsilon, dampingfactor, maxiter);
#endif
#endif

#if ADD_GRAPH_UPDATE == 1
  update_time_trace = xmalloc (nbatch * sizeof(*update_time_trace));

  /* Convert to STINGER */
  tic ();
  //S = stinger_new ();
  //stinger_set_initial_edges (S, nv, 0, off, ind, weight, NULL, NULL, -2);
  PRINT_STAT_DOUBLE ("time_stinger", toc ());
  fflush(stdout);

  free(graphmem);

  tic ();
  uint32_t errorCode = stinger_consistency_check (S, nv);
  double time_check = toc ();
  PRINT_STAT_HEX64 ("error_code", (long unsigned) errorCode);
  PRINT_STAT_DOUBLE ("time_check", time_check);

  /* Updates */
  int64_t ntrace = 0;

#if 0
  stinger_insert_edge (S, 0, 1, 2, 1, 2);
  stinger_insert_edge (S, 0, 2, 1, 1, 2);
#else

  for (int64_t actno = 0; actno < nbatch * batch_size; actno += batch_size)
  {
    tic();
    const int64_t endact = (actno + batch_size > naction ? naction : actno + batch_size);
    int64_t *actions = &action[2*actno];
    int64_t numActions = endact - actno;

//    MTA("mta assert parallel")
//    MTA("mta block dynamic schedule")
//    OMP("omp parallel for")
//	for(uint64_t k = 0; k < endact - actno; k++) {
	for(uint64_t k = 0; k < 1; k++) {
	  printf("----k: %" PRId64 "\n", k);
	  //printf("----endact - actno: %" PRId64 "\n", endact - actno);
	  //const int64_t i = actions[2 * k];
      //const int64_t j = actions[2 * k + 1];
	  int64_t i = actions[2 * k];
      int64_t j = actions[2 * k + 1];
      //printf("----i: %" PRId64 "\n", i);
     // printf("----j: %" PRId64 "\n", j);

      //stinger/stinger_core/src/stinger.c
      //because of --> nv    = (3*nv)/4;
      if (i > nv * 0.75) i = nv * 0.7;
      if (j > nv * 0.75) j = nv * 0.7;
#if 1
      if (i != j && i < 0) {
	     stinger_remove_edge(S, 0, ~i, ~j);
	     stinger_remove_edge(S, 0, ~j, ~i);
      }
#endif
      if (i != j && i >= 0) {
	     stinger_insert_edge (S, 0, i, j, 1, actno+2);
	     stinger_insert_edge (S, 0, j, i, 1, actno+2);
      }

    }
//    update_time_trace[ntrace] = toc();
//    ntrace++;
  } /* End of batch */
#endif

#if 0
  /* Print the times */
  double time_updates = 0;
  for (int64_t k = 0; k < nbatch; k++) {
    time_updates += update_time_trace[k];
  }
  PRINT_STAT_DOUBLE ("time_updates", time_updates);
  PRINT_STAT_DOUBLE ("updates_per_sec", (nbatch * batch_size) / time_updates);

  tic ();
  errorCode = stinger_consistency_check (S, nv);
  time_check = toc ();
  PRINT_STAT_HEX64 ("error_code", (long unsigned) errorCode);
  PRINT_STAT_DOUBLE ("time_check", time_check);

  free(update_time_trace);
#endif
//  stinger_free_all (S);
//  free (actionmem);
#endif //end of update


#if 1
  //initialize pr, tmp_pr
  free(tmp_pr);
  free(pr);
  pr = (double*)xmalloc(sizeof(double) * nv);
  tmp_pr = (double*)xmalloc(sizeof(double) * nv);
  //call pagerank again
#if TEST_EPSLION_THRESHOLD == 1
  LOG_I_A("threshold: %5.5e, epsilon: %5.5e", threshold, epsilonchange);
  page_rank_directed(S, nv, pr, tmp_pr, epsilonchange, dampingfactor, maxiter, threshold);
#else
  page_rank_directed(S, nv, pr, tmp_pr, epsilon, dampingfactor, maxiter);
#endif
#endif

#if COUNT_NUM_V_DELTA == 1
  fclose(fp);
#endif
  STATS_END ();
  free(tmp_pr);
  free(pr);

#if SAVE_PR_VALUES == 1
//pr value comparison
  fp =    fopen("pr_values_0.txt", "r");
  fp_pr = fopen("pr_values_1.txt", "r");

  if (fp == NULL || fp_pr == NULL)
  {
	  printf("ERROR: files are not exists!\n");
	  exit(1);
  }

  double val1, val2;
  double diff_epsilon = 1e-8;
  int diff_count = 0;
  //int num = 0;
  while (!feof(fp) && !feof(fp_pr))
  //if (num == 0)
  {

	  fscanf(fp, "%lf", &val1);
	  fscanf(fp_pr, "%lf", &val2);
	  //printf("val1: %e, val2: %e\n", val1, val2);
	  //printf("val1-val2: %e\n", fabs(val1-val2));
	  if (fabs(val1-val2) > diff_epsilon) diff_count++;
#if 0
	  printf("val1: %e, val2: %e\n", val1, val2);
	  printf("num: %d\n", num++);
//#else
	  diff_count++;
	  printf("diff_count: %d\n", diff_count);
#endif
  }
  printf("diff_count: %d\n", diff_count);
  fclose(fp);
#endif
}


inline double * set_tmp_pr(double * tmp_pr_in, int64_t NV) {
  double * tmp_pr = NULL;
  if (tmp_pr_in) {
    tmp_pr = tmp_pr_in;
  } else {
    tmp_pr = (double *)xmalloc(sizeof(double) * NV);
  }
  return tmp_pr;
}

inline void unset_tmp_pr(double * tmp_pr, double * tmp_pr_in) {
  if (!tmp_pr_in)
    free(tmp_pr);
}
#if 0
int64_t
page_rank_subset(stinger_t * S, int64_t NV, uint8_t * vertex_set, int64_t vertex_set_size, double * pr, double * tmp_pr_in, double epsilon, double dampingfactor, int64_t maxiter) {
  double * tmp_pr = set_tmp_pr(tmp_pr_in, NV);

  int64_t * vtx_outdegree = (int64_t *)xcalloc(NV,sizeof(int64_t));

  OMP("omp parallel for")
  for (uint64_t v = 0; v < NV; v++) {
    if (vertex_set[v]) {
      STINGER_FORALL_EDGES_OF_VTX_BEGIN(S,v) {
        if (vertex_set[STINGER_EDGE_DEST]) {
          vtx_outdegree[v]++;
        }
      } STINGER_FORALL_EDGES_OF_VTX_END();
    }
  }

  int64_t * pr_lock = (int64_t *)xcalloc(NV,sizeof(double));

  int64_t iter = maxiter;
  double delta = 1;
  int64_t iter_count = 0;

  while (delta > epsilon && iter > 0) {
    iter_count++;

    double pr_constant = 0.0;

    OMP("omp parallel for reduction(+:pr_constant)")
    for (uint64_t v = 0; v < NV; v++) {
      tmp_pr[v] = 0;
      if (vtx_outdegree[v] == 0) {
        pr_constant += pr[v];
      }
    }

    STINGER_PARALLEL_FORALL_EDGES_OF_ALL_TYPES_BEGIN(S) {
      if (vertex_set[STINGER_EDGE_DEST] && vertex_set[STINGER_EDGE_SOURCE]) {
        int64_t outdegree = vtx_outdegree[STINGER_EDGE_SOURCE];
        int64_t count = readfe(&pr_lock[STINGER_EDGE_DEST]);
        tmp_pr[STINGER_EDGE_DEST] += (((double)pr[STINGER_EDGE_SOURCE]) /
          ((double) outdegree));
        writeef(&pr_lock[STINGER_EDGE_DEST],count+1);
      }
    } STINGER_PARALLEL_FORALL_EDGES_OF_ALL_TYPES_END();

    OMP("omp parallel for")
    for (uint64_t v = 0; v < NV; v++) {
      if (vertex_set[v]) {
        tmp_pr[v] = (tmp_pr[v] + pr_constant / (double)vertex_set_size) * dampingfactor + (((double)(1-dampingfactor)) / ((double)vertex_set_size));
      }
    }

    delta = 0;
    OMP("omp parallel for reduction(+:delta)")
    for (uint64_t v = 0; v < NV; v++) {
      if (vertex_set[v]) {
        double mydelta = tmp_pr[v] - pr[v];
        if (mydelta < 0)
          mydelta = -mydelta;
        delta += mydelta;
      }
    }
    //LOG_I_A("delta : %20.15e", delta);

    OMP("omp parallel for")
    for (uint64_t v = 0; v < NV; v++) {
      if (vertex_set[v]) {
        pr[v] = tmp_pr[v];
      }
    }

    iter--;
  }

  unset_tmp_pr(tmp_pr,tmp_pr_in);
  xfree(pr_lock);
  xfree(vtx_outdegree);
}
#endif

int64_t
#if TEST_EPSLION_THRESHOLD == 1
page_rank_directed(stinger_t * S, int64_t NV, double * pr, double * tmp_pr, double epsilon, double dampingfactor, int64_t maxiter, double threshold) {
#else
page_rank_directed(stinger_t * S, int64_t NV, double * pr, double * tmp_pr, double epsilon, double dampingfactor, int64_t maxiter) {
#endif
 // double * tmp_pr = set_tmp_pr(tmp_pr_in, NV);

  int64_t * pr_lock = (int64_t *)xcalloc(NV,sizeof(double));

  for(uint64_t v = 0; v < NV; v++) { //pr[] initialization
    pr[v] = 1 / ((double)NV);
  }

  int64_t iter = maxiter;
  double delta = 1;
  int64_t iter_count = 0;

#if SAVE_PR_VALUES == 1
  LOG_I_A("!!!!!!!!!!!!!!!!called %d \n", fp_idx);
  char buff[50];
  sprintf(buff, "pr_values_%d.txt", fp_idx++);
  fp_pr = fopen(buff, "w");
#endif

#if TEST_EPSLION_THRESHOLD == 1
  LOG_I_A("NV : %ld", NV);
  LOG_I_A("threshold: %5.10e, epsilon: %5.10e", threshold, epsilon);
#endif

#if NO_ITERATION_LIMIT == 1
  while (delta > epsilon) { //no limit for iteration
#else
  while (delta > epsilon && iter > 0) {
#endif

    iter_count++;
    double pr_constant = 0.0;

    printf("iter_count: %d\n", iter_count);

//    OMP("omp parallel for reduction(+:pr_constant)")
    for (uint64_t v = 0; v < NV; v++) {
      tmp_pr[v] = 0;
      if (stinger_outdegree(S,v) == 0) {
#if 0
        if (v == NV - 1)
        {
          printf("pr[%ld]:%5.5e, ", v, pr[v]);
        }
#endif
        pr_constant += pr[v];
#if 0
        if (v == NV - 1)
        {
       	  printf("pr_constant: %5.5e\n", pr_constant);
        }
#endif
      }
    }
    //LOG_I_A("pr_constant : %20.15e", pr_constant);

    STINGER_PARALLEL_FORALL_EDGES_OF_ALL_TYPES_BEGIN(S) {
      int64_t outdegree = stinger_outdegree(S, STINGER_EDGE_SOURCE);
      int64_t count = readfe(&pr_lock[STINGER_EDGE_DEST]);
      tmp_pr[STINGER_EDGE_DEST] += (((double)pr[STINGER_EDGE_SOURCE]) /
        ((double) (outdegree ? outdegree : NV -1)));
      writeef(&pr_lock[STINGER_EDGE_DEST],count+1);
    } STINGER_PARALLEL_FORALL_EDGES_OF_ALL_TYPES_END();

#if 0
 if (iter_count == 10)
 {
	 for (uint64_t v = 1000; v < 1010; v++) {
		printf("tmp_pr[%d]  = %5.5e\n", v, tmp_pr[v]);
		printf("pr_lock[%d] = %5.5e\n", v, pr_lock[v]);
	 }
 }
#endif

//    OMP("omp parallel for")
    for (uint64_t v = 0; v < NV; v++) {

      if (v == 0)
      {
    	  printf("tmp_pr[%d]  = %5.5e, pr_constant: %5.5e, pr_constant / (double)NV: %5.5e\n", v, tmp_pr[v], pr_constant, pr_constant / (double)NV);
      }
      tmp_pr[v] = (tmp_pr[v] + pr_constant / (double)NV) * dampingfactor + (((double)(1-dampingfactor)) / ((double)NV));
      //tmp_pr[v] = (tmp_pr[v]) * dampingfactor + (((double)(1-dampingfactor)) / ((double)NV));
    }
#if COUNT_NUM_V_DELTA == 1
    delta = 0;
    int count = 0;
#endif

//    OMP("omp parallel for reduction(+:delta)")
    for (uint64_t v = 0; v < NV; v++) {
      double mydelta = tmp_pr[v] - pr[v];
      if (mydelta < 0)
        mydelta = -mydelta;
      delta += mydelta;
      double change = mydelta/pr[v];
#if COUNT_NUM_V_DELTA == 1
      if ( change > threshold ) count++;
#endif
    }
#if COUNT_NUM_V_DELTA == 1
    //LOG_I_A("delta : %20.15e", delta);
    //LOG_I_A("count : %d", count);
    fprintf(fp, "%d ", count);
#endif

//    OMP("omp parallel for")
    for (uint64_t v = 0; v < NV; v++) {
      pr[v] = tmp_pr[v];
    }
//    LOG_I_A("iter : %ld", iter);
#if NO_ITERATION_LIMIT != 1
    iter--; //no limit for iteration
#endif
  }
#if COUNT_NUM_V_DELTA == 1
  fprintf(fp, "\n");
#endif

#if SAVE_PR_VALUES == 1
  for (uint64_t v = 0; v < NV; v++) {
    fprintf(fp_pr, "%e ", pr[v]);
    if (((v+1) % 10) == 0) fprintf(fp_pr, "\n");
  }
  fclose(fp_pr);
#endif
  //unset_tmp_pr(tmp_pr,tmp_pr_in);
  xfree(pr_lock);
}

// NOTE: This only works on Undirected Graphs!
int64_t
page_rank (stinger_t * S, int64_t NV, double * pr, double * tmp_pr_in, double epsilon, double dampingfactor, int64_t maxiter)
{
  double * tmp_pr = set_tmp_pr(tmp_pr_in, NV);
  int64_t iter = maxiter;
  double delta = 1;
  int64_t iter_count = 0;

  while (delta > epsilon && iter > 0) {
    iter_count++;

    double pr_constant = 0.0;

    OMP("omp parallel for reduction(+:pr_constant)")
    for (uint64_t v = 0; v < NV; v++) {
      tmp_pr[v] = 0;
    }

    OMP("omp parallel for")
    for (uint64_t v = 0; v < NV; v++) {
      tmp_pr[v] = (tmp_pr[v] + pr_constant / (double)NV) * dampingfactor + (((double)(1-dampingfactor)) / ((double)NV));
    }

    delta = 0;
    OMP("omp parallel for reduction(+:delta)")
    for (uint64_t v = 0; v < NV; v++) {
      double mydelta = tmp_pr[v] - pr[v];
      if (mydelta < 0)
        mydelta = -mydelta;
        delta += mydelta;
    }

    OMP("omp parallel for")
    for (uint64_t v = 0; v < NV; v++) {
      pr[v] = tmp_pr[v];
    }

    iter--;
  }

  LOG_I_A("PageRank iteration count : %ld", iter_count);

  unset_tmp_pr(tmp_pr,tmp_pr_in);
}
