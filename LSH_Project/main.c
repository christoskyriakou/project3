#define _POSIX_C_SOURCE 200809L // Ενεργοποιεί τα POSIX features , έπαιρνα warning για την getopt πάρα το ότι είχα την #include <unistd.h> και αυτό το έφτιαξε
#include "lsh.h"
#include <string.h>
#include <time.h>
#include <strings.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <getopt.h>
double total_af = 0.0;
int L=5;    
int K=4;
double W=4.0;
int N=1;     
double R=2.0;
int SEED=1;

double randn(){ // Λειτουργεί με βάση το Central Limit Theorem επιστέφει 
    double sum=0.0; 
    for(int i=0; i<12;i++){
        sum+=(double)rand()/RAND_MAX; // rand() επιστρέφει ακέραιο από 0 έως RAND_MAΧ και διαιρούμε με RAND_MAX έχουμε αριθμό στο [0,1]
    }
    return sum-6.0; // Μετατροπή για να προκύψει περίπου κανονική κατανομή με μέση τιμή 0
}


int32_t be_to_le(int32_t number){ // Μετατροπή από big endian σε little επειδή είμαστε σε linux περιβάλλον ubuntu
    return(((number & 0xFF000000)>>24)|((number & 0x00FF0000)>>8)|  // Με τα shifts αυτα μεταφέρουμε το κάθε byte στην αντίστοιχη θέση και μετά με το or φτιάχνουμε τον le αριθμό 
        ((number & 0x0000FF00)<<8)|((number & 0x000000FF)<<24));
}

int load_mnist_images(const char* filename, double**dataset, int *num_images,int *dim_out ){ 
    FILE *f=fopen(filename,"rb"); // Ανοίγουμε το binary αρχείο αν δεν ανοίξει επιστρέφεται σφάλμα
    if(!f){ 
    perror("ERROR when opening the MNSIT file");
    exit(1);
    }
    int32_t magic=0; // αρχικοποιούμε τις παραμέτρους του mnsit που ειναι 32bitες 
    int32_t n_rows=0;
    int32_t n_cols=0;
    if(fread(&magic,4,1,f)!=1||fread(num_images,4,1,f)!=1||fread(&n_rows,4,1,f)!=1||fread(&n_cols,4,1,f)!=1){ //Αν αποτύχουν οι freads για τις παραμέτρους επιστρέφεται σφάλμα
        perror("ERROR when reading the MSIT file");
        exit(1);
    }
    magic=be_to_le(magic); // Μετατρέπουμε αφού τα διαβάσαμε τις παραμέτρους σε little endian!
    *num_images=be_to_le(*num_images); 
    n_rows=be_to_le(n_rows); 
    n_cols=be_to_le(n_cols);
    if(magic!=2051){ // Αν ειναι λάθος ο μαγικός αριθμός τότε επιστρέφεται σφάλμα 
        perror("Magic number is wrong"); 
        exit(1); 
    } 
    size_t total_pixels=(size_t)(*num_images)*n_rows*n_cols; // το σύνολο των pixels ολου του αρχείο 
    uint8_t *temp_data=malloc(total_pixels*sizeof(uint8_t)); // Δεσμέυουμε μνήμη για τα προσωρινά δεδομένα 
    if (temp_data==NULL){  // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα
        perror(" ERROR! malloc failed for temp_data");
        fclose(f);
        exit(1); 
    }
    if(fread(temp_data,1,total_pixels, f)!=total_pixels) {  // Διαβάζουμε όλα τα pixels από το αρχείο στο προσωρινό buffer
        perror("Error reading all pixel data");
        free(temp_data);
        fclose(f);
        exit(1); 
    }
    *dataset=malloc(total_pixels*sizeof(double)); // Δέσμευση μνήμης για το τελικό dataset σε τύπο double
    if (!*dataset){ 
        perror(" ERROR! malloc failed for final dataset");
        free(temp_data);
        fclose(f);
        exit(1); 
    }
    for (size_t i=0;i<total_pixels;i++){ // Μετατρέπουμε κάθε pixel από uint8_t σε double
    (*dataset)[i]=(double)temp_data[i]; 
    }
    free(temp_data); // Ελευθερώνουμε την μνήμη που δεσμεύτηκε προσωρινα
    int D=n_rows*n_cols; // Ορισμός της διάστασης κάθε εικόνας
    *dim_out=D;
    fclose(f);  // Κλείσιμο του αρχείου
    return(0);
}

int compareNeighbors(const void* a,const void* b){ // συγκρίνο
    const NearestNeighbor* element_a=(const NearestNeighbor*)a; // Μετατρέπουμε τα void pointers σε pointers προς NearestNeighbor
    const NearestNeighbor* element_b=(const NearestNeighbor*)b;
    if(element_a->distance<element_b->distance){ // Σύγκριση των αποστάσεων για ταξινόμηση και επσιτροφή -1 αν το α ειναι μικροτερο, 1 αν το ειναι το β μικροτερο και 0 αν ειναι ίσες αποστάσεις
        return -1;
    }
    else if(element_a->distance > element_b->distance){
        return 1;
    }
    return 0;
}
int load_sift_vectors(const char* filename,double** dataset,int *num_vectors,int *dim_out) {
    FILE *f=fopen(filename,"rb"); // Ανοίγουμε το binary αρχείο αν δεν ανοίξει επιστρέφεται σφάλμα
    if(!f){ 
        perror("ERROR when opening the SIFT file");
        return 1;
    }
    int32_t dim=0; 
    // 1. Διαβάζουμε το dim (4 bytes) από την αρχή
    if(fread(&dim,sizeof(int32_t),1,f)!=1){ //Διαβάζουμε την πρώτη τιμή (4 bytes) που αντιστοιχεί στη διάσταση κάθε διανύσματος SIFT
        perror("Error reading first SIFT dimension");
        fclose(f);
        return 1;
    }
    *dim_out=dim; // θα πρέπει να ειναι 128
    fseek(f,0,SEEK_END); // Πηγαίνουμε στο τέλος του αρχείου και μετράμε το συνολικό μέγεθος του αρχείου σε bytes
    long file_size=ftell(f); // Μέγεθος αρχείου
    size_t vector_size_bytes=sizeof(int32_t)+(size_t)dim*sizeof(float);
    int total_count=(int)(file_size/vector_size_bytes);   // Υπολογισμός συνολικού πλήθους διανυσμάτων στο αρχείο
    *num_vectors=total_count;
    printf("The Sift file has: Vectors: %d,Dimension: %d\n",*num_vectors,*dim_out);
    size_t total_doubles=(size_t)total_count*dim; // Επιστροφή του πλήθους στον καλούντα
    *dataset=(double*)malloc(total_doubles*sizeof(double));// Δέσμευση μνήμης για το dataset που θα περιέχει όλα τα διανύσματα σε double μορφή
    if(dataset==NULL){  // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα
        perror("ERROR! malloc failed for SIFT dataset");
        fclose(f);
        return 1; 
    }
    fseek(f,0,SEEK_SET);// Επιστρέχουμε τον δείκτη ανάγνωσης στην αρχή του αρχείου
    for(int i=0;i<total_count;i++){   // Επαναληπτική ανάγνωση όλων των διανυσμάτων SIFT
        int32_t current_dim=0; // Κάθε διάνυσμα ξεκινά με μία 4-byte τιμή που επαναλαμβάνει τη διάσταση
        if(fread(&current_dim,sizeof(int32_t),1,f)!=1){
             perror("Error reading SIFT dim before vector data");
             free(*dataset);
             fclose(f);
             return 1;
        }
        for(int j=0;j<dim;j++){ // ΕΠαναληπτική ανάγνωση τις float τιμές του διανύσματος
            float f_val; 
            if (fread(&f_val,sizeof(float),1,f)!=1){
                perror("Error reading SIFT float value");
                free(*dataset);
                fclose(f);
                return 1;
            }
            (*dataset)[i*dim+j]=(double)f_val; // Κάνουμε τα floats, doubles ώστε να μπορουν να χρησιμοποιήθουν κοινές συναρτήσεις με τα mnsit
        }
    }
    fclose(f); //Κλείνουμε το αρχείο
    return 0;
}

NearestNeighbor* BruteForceSearch(const double* dataset,int num_vectors,int dim,const double* query,int N,double R,int* num_R_neighbors,int** R_neighbors_indices){ 
    NearestNeighbor* nn_list=malloc(sizeof(NearestNeighbor)*num_vectors); // Δεσμεύουμε μνήμη για τον πίνακα που θα περιέχει όλους τους υποψήφιους γείτονες.
    if(nn_list==NULL){ // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα
        perror("malloc failed");   
        exit(1);
    }
    for(int f=0;f<num_vectors;f++){ // Για κάθε διάνυσμα του dataset, υπολογίζουμε την ευκλείδεια απόσταση από το query.
        const double* current_vector=&dataset[f*dim]; // Δείχνουμε στο f-οστό διάνυσμα και υπολογίζουμε την απόσταση.
        double dist=euclidean_distance(query,current_vector,dim);
        nn_list[f].index=f; // Αποθηκεύουμε τα στοιχεία του
        nn_list[f].distance=dist;
    }
    qsort(nn_list,num_vectors,sizeof(NearestNeighbor),compareNeighbors); // Ταξινομούμε όλα τα διανύσματα με βάση την απόσταση (αύξουσα σειρά).
    NearestNeighbor* nn_list_N=malloc(sizeof(NearestNeighbor)*N);  // Δεσμεύουμε μνήμη για τα Ν κοντινότερα γειτονικά διανύσματα που θα επιστραφούν.
    if(nn_list_N==NULL){ // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα
        perror("malloc failed"); 
        exit(1);
    }
    for(int k=0;k<N;k++){  // Αντιγράφουμε τα Ν πρώτα στοιχεία από τον πλήρως ταξινομημένο πίνακα.
        nn_list_N[k]=nn_list[k];
    }
    int* R_nn_list=malloc(sizeof(int)*num_vectors); // Δεσμεύουμε μνήμη για τους δείκτες των γειτόνων εντός της ακτίνας R.
    if(R_nn_list==NULL){ // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα
        perror("malloc failed"); 
        exit(1);
    }
    int R_counter=0; // Συλλέγουμε τους δείκτες των διανυσμάτων που βρίσκονται εντός της ακτίνας R.
    for(int p=0;p<num_vectors;p++){
        if (nn_list[p].distance<=R){
            R_nn_list[R_counter++]=nn_list[p].index;
        }
    }
    *num_R_neighbors=R_counter;// Επιστρέφουμε στον καλούντα τον αριθμό και τους δείκτες των γειτόνων εντός R.
    *R_neighbors_indices=R_nn_list;
    free(nn_list);  // Αποδεσμεύουμε τον προσωρινό πίνακα των υποψήφιων γειτόνων.
    return nn_list_N;   // Επιστρέφουμε τον πίνακα με τα Ν πλησιέστερα γειτονικά διανύσματα.
}

double calculate_recall(const NearestNeighbor* lsh_nn,const NearestNeighbor* true_nn,int N){
    int hits=0;
    int true_indices[N];   // Δημιουργούμε έναν πίνακα για γρήγορο έλεγχο των N σωστών indices από το true_nn
    for(int i=0;i<N;i++){
        true_indices[i]=true_nn[i].index;  // Αποθηκεύουμε τους δείκτες των πραγματικών πλησιέστερων γειτόνων
    }
    for(int i=0;i<N;i++){  // Ελέγχουμε πόσοι τους N γείτονες του LSH είναι σωστοί
        int lsh_index=lsh_nn[i].index;
        if(lsh_index<0||lsh_nn[i].distance>=INFINITY) continue; // Αγνοούμε μη-έγκυρους
        for(int j=0;j<N;j++){
            if (lsh_index==true_indices[j]){
                hits++;
                break; // Βρέθηκε ο σωστός, προχωράμε στον επόμενο LSH γείτονα
            }
        }
    }
    return (double)hits; // Επιστρέφουμε τον αριθμό των hits ως double
}

int main(int argc,char*argv[]){
    char *train_file_path=NULL; // Aρχικοποιούμε τα filenames και το είδος του αρχείου εισαγωγής
    char *query_file_path=NULL;
    char *output_file_path=NULL;
    int data_type=DATA_TYPE_SIFT; // 
    int use_range=0 ;// Για το range
    int use_lsh = 0;  // για το --lsh
    for(int i=1;i<argc;++i){
    if(strcmp(argv[i],"-d")==0&&i+1<argc) train_file_path=argv[++i];
    else if(strcmp(argv[i],"-q")==0&&i+1<argc) query_file_path=argv[++i];
    else if(strcmp(argv[i],"-k")==0&&i+1<argc) K=atoi(argv[++i]);
    else if(strcmp(argv[i],"-L")==0&&i+1<argc) L=atoi(argv[++i]);
    else if(strcmp(argv[i],"-w")==0&&i+1<argc) W=atof(argv[++i]);
    else if(strcmp(argv[i],"-N")==0&&i+1<argc) N=atoi(argv[++i]);
    else if(strcmp(argv[i],"-R")==0&&i+1<argc) R=atof(argv[++i]);
    else if(strcmp(argv[i],"-o")==0&&i+1<argc) output_file_path=argv[++i];
    else if(strcmp(argv[i],"-type")==0&&i+1<argc){
        if(strcasecmp(argv[i+1],"sift")==0) data_type=DATA_TYPE_SIFT;
        else if(strcasecmp(argv[i+1],"mnist")==0) data_type=DATA_TYPE_MNIST;
        else{fprintf(stderr,"Unknown  dataset type: %s\n",argv[i+1]); 
            exit(1);
        }
        i++; 
    }
    else if(strcmp(argv[i],"-range")==0&&i+1<argc){
        if(strcasecmp(argv[i+1],"true")==0) use_range=1;
        else if(strcasecmp(argv[i+1],"false")==0) use_range=0;
        else{fprintf(stderr,"Unknown range value: %s\n",argv[i+1]); 
            exit(1);}
        i++;
    }
    else if(strcmp(argv[i],"-lsh")==0) use_lsh=1;
    else{fprintf(stderr,"Unknown or incomplete option: %s\n",argv[i]); exit(1);}
    }
    if(!train_file_path||!query_file_path||!output_file_path){ //ελέχουμε ότι τα πεδιά των αρχείων δεν είναι κενά 
        fprintf(stderr,"Files not included");
        exit(1);
    }
    srand(SEED); //Παιρνάμε το φύτρο των ψευδοτυχαιών 
    double *train_dataset=NULL;
    int num_train=0,dim=0;
    double *query_dataset=NULL;
    int num_query=0,query_dim=0;
    int status; // Ελέγχουμε ότι φορτώθηκαν σωστά τα αρχεία των vectors 
    status=(data_type==DATA_TYPE_MNIST)?load_mnist_images(train_file_path,&train_dataset,&num_train,&dim):load_sift_vectors(train_file_path,&train_dataset,&num_train,&dim);
    if(status!=0){
        printf("error\n"); 
        return 1;
    }
    status=(data_type==DATA_TYPE_MNIST)?load_mnist_images(query_file_path,&query_dataset,&num_query,&query_dim):load_sift_vectors(query_file_path,&query_dataset,&num_query,&query_dim);
    if(status!=0){
        printf("error\n"); 
        return 1;
    }
    if(dim!=query_dim){ // Αν οι διαστάσεις των διανυσμάτων είναι λάθος επιστρέφουμε σφαλμα και αποδεσμεύουμε την μνήμη 
        printf("error\n"); 
        free(train_dataset); 
        free(query_dataset);
         return 1;}
    const char *method_name="LSH"; // To όνομα της μεθόδου
    FILE *out=fopen(output_file_path,"w"); // Ανοίγουμε το αρχείο για εγγραφή σε αυτό 
    if(!out){ // Αν δεν ανοίξει error
        printf("error\n"); 
        return 1;
    }
    hashtable *lsh_tables=malloc(sizeof(hashtable)*L); // δεσμεύουμε μνήμη για τα L hastables , τα αρχικοποιούμε και βάζουμε σε αυτά τα vectors
    for(int i=0;i<L;++i){
        initHashTable(&lsh_tables[i],dim);
    }
    for(int i=0;i<num_train;++i){
        const double *vec=&train_dataset[(size_t)i*dim];
        for(int j=0;j<L;++j) insertVector(&lsh_tables[j],vec,i,dim);
    }
    double total_recall_hits=0.0, total_lsh_time=0.0, total_brute_time=0.0, total_af_sum=0.0;  // Ορίζουμε τις μεταβλητές για τα results
    int max_queries=(num_query>100)?100:num_query; // Μεγιστος αριθμός που θα εξτάσουμε ειναι 100 
    for(int q=0;q<max_queries;++q){// Για max_queries φορές, θα κάνουμε σύγκρισεις
        const double *qvec=&query_dataset[(size_t)q*dim];//  αντλούμε το query από το trainset
        NearestNeighbor *lsh_nn = NULL;  
        if (use_lsh) {
        clock_t ta=clock();//Αρχιζουμε την χρονομέτρηση για το lsh nn search και το εκετελούμε 
        lsh_nn=find_l_nn(lsh_tables,qvec,train_dataset,num_train,N,dim);
        clock_t tb=clock();  // Πα΄ρινουμε τον επόμενο χρόνο, την στιγμή λήξης και βρίσκουμε την διαφορά τους και την διαιρούμε με την μονάδα του χρόνου ( 1 ρολόι)
        double lsh_time=(double)(tb-ta)/CLOCKS_PER_SEC; 
        total_lsh_time+=lsh_time;// Το προσθέτουμε στον συνολικό χρόνο 
        }
        int num_R_neighbors=0,*R_indices=NULL;  // Αρχικοποιούμε τις μεταβλητές για το range search
        clock_t tc=clock(); // Κάνουμε το ίδιο για τον χρόνο του brute force.
        NearestNeighbor *true_nn=BruteForceSearch(train_dataset,num_train,dim,qvec,N,R,&num_R_neighbors,&R_indices);
        clock_t td=clock();
        double brute_time=(double)(td-tc)/CLOCKS_PER_SEC;
        total_brute_time+=brute_time;

        double hits=calculate_recall(lsh_nn,true_nn,N); // Υπολογίζουμε το recall για το διάνυσμ και το προσθέτουμε στα συνολικά hits
        total_recall_hits += hits;
        double query_af_sum=0.0,query_af_count=0; // Αρχικοποιούμε δύο μεταβλητές για να υπολογίσουμε το AF για το τρέχον query
        for(int i=0;i<N;++i){
            double true_d=(i<N)?true_nn[i].distance:DBL_MAX;
            double approx_d=(lsh_nn&&i<N)?lsh_nn[i].distance:DBL_MAX;
          if(true_d !=0.0&&approx_d<DBL_MAX){ // Αν η πραγματική αποσταση δεν εινα 0 και ειναι μικρόυερη από την μέγιστη ορισμένη τιμή τότε υπολογίζεται το af
                query_af_sum+=approx_d/true_d;
                query_af_count++;
            }
        }
        if(query_af_count>0){ 
            total_af_sum+=query_af_sum/query_af_count; 
        }
        fprintf(out,"%s\n",method_name);
        fprintf(out,"Query: %d\n",q);
        for(int i=0;i<N;++i){ // Για τα Ν διανύσματα, αρχικοποιούμε το nn index με default τιμές ( μη ευρεσης)
            int nn_index=-1;
            double approx_d=DBL_MAX,true_d=DBL_MAX; 
            if(lsh_nn && i<N){  // Μέχρι να φτάσουμε τους Ν γείτονες, παιρνάμε από τα αποτελσματα του lsh nn και τα γράφουμε σύμφωνα με τις προδιαγραφές της υπόθεσης
                nn_index=lsh_nn[i].index; 
                approx_d=lsh_nn[i].distance;
            }
            if(true_nn && i<N){
                true_d=true_nn[i].distance;
            }
            fprintf(out,"Nearest neighbor-%d: %d\n",i+1,nn_index);
            fprintf(out,"distanceApproximate: %.6f\n",approx_d);
            fprintf(out,"distanceTrue: %.6f\n",true_d);
        }
        fprintf(out,"R-near neighbors:\n");
        if(use_range){ // Αν έχουμε ενεργοποιήσει το range search, σε περιπτωση που δεν βρεθεί εικόνα επιστρέφετα μηνυμα, ειδάλλως γράφονται τα διανύσματα του R indices
            if(num_R_neighbors<=0) fprintf(out,"Καμία εικόνα\n");
            else for(int r=0;r<num_R_neighbors;++r) fprintf(out,"%d\n",R_indices[r]);
        }else fprintf(out,"Range Search Disabled\n");
        fprintf(out,"\n"); // Απελευθερώνουμε την μνήμη αφού κάναμε τις εγγραφες
        if(lsh_nn){ free(lsh_nn);}
        if(true_nn) {free(true_nn);}
        if(R_indices) {free(R_indices);} 
    }
    double total_possible=(double)(max_queries*N);// Υπολογίζουμε τους μέσους όρους των αποτελεσμάτων για όλα τα queries
    double avg_recall=(total_possible>0.0)?(total_recall_hits/total_possible):0.0;
    double avg_lsh_time=(max_queries>0)?total_lsh_time/max_queries:0.0;
    double avg_brute_time=(max_queries>0)?total_brute_time/max_queries:0.0;
    double avg_af=(max_queries>0)?total_af_sum/max_queries:0.0;
    double qps=(avg_lsh_time>1e-12)?1.0/avg_lsh_time:0.0;

    fprintf(out,"Average AF: %.4f\n",avg_af); // Tους περνάμε στο αρχείο
    fprintf(out,"Recall@%d: %.4f\n",N,avg_recall);
    fprintf(out,"QPS: %.2f\n",qps);
    fprintf(out,"tApproximateAverage: %.6f sec\n",avg_lsh_time);
    fprintf(out,"tTrueAverage: %.6f sec\n",avg_brute_time);
    fclose(out);
    for(int i=0;i<L;++i){  // Αποδεσμέυουμε την μνήμη και επιστρέφουμε μύνημα επιτυχίας
        freeHashTable(&lsh_tables[i]);
    }
    free(lsh_tables);
    free(train_dataset);
    free(query_dataset);
    printf("Success!");
    return 0;
}
