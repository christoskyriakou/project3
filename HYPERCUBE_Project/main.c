#include "hc.h"
#include <string.h>
#include <unistd.h>
#include <time.h>

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
#define DATA_TYPE_MNIST 0
#define DATA_TYPE_SIFT 1
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

char* output_file_path=NULL; // Δήλωση της εξωτερικής μεταβλητής για το όνομα του αρχείου εξόδου
int main(int argc,char *argv[]){
    char *train_file_path=NULL;
    char *query_file_path=NULL;
    char *output_file=NULL;
    int K_PROJ = 14; // Default τιμές των arguments βάσει των προδιαγραφών
    int N_NEIGHBORS=1;
    double R_RADIUS=2.0;
    double W=4.0;
    int SEED=1;
    int MAX_CANDIDATES=DEFAULT_M;
    int PROBES=DEFAULT_PROBES;
    int data_type=DATA_TYPE_MNIST;
    int use_hypercube=1; // Έλεγχος για την χρήση hypercube ( Απλά για να υπακούει στις προδιαγραφές)
    int use_range=1; // Default τιμή για την use range, ο χρήστης με το input μπορει να το αλλάξει.
    int opt; // Θα μας βοηθήσει με την getopt να διακρίνουμε τις περιπτώσεις των arguments
    for (int i=1;i<argc;i++) {
    if (strcmp(argv[i],"-d")==0 && i+1<argc)
        train_file_path = argv[++i];
    else if (strcmp(argv[i],"-q")==0 && i+1<argc)
        query_file_path = argv[++i];
    else if (strcmp(argv[i],"-kproj")==0 && i+1<argc)
        K_PROJ = atoi(argv[++i]);
    else if (strcmp(argv[i],"-w")==0 && i+1<argc)
        W = atof(argv[++i]);
    else if (strcmp(argv[i],"-M")==0 && i+1<argc)
        MAX_CANDIDATES = atoi(argv[++i]);
    else if (strcmp(argv[i],"-probes")==0 && i+1<argc)
        PROBES = atoi(argv[++i]);
    else if (strcmp(argv[i],"-o")==0 && i+1<argc)
        output_file = argv[++i];
    else if (strcmp(argv[i],"-N")==0 && i+1<argc)
        N_NEIGHBORS = atoi(argv[++i]);
    else if (strcmp(argv[i],"-R")==0 && i+1<argc)
        R_RADIUS = atof(argv[++i]);
    else if (strcmp(argv[i],"-type")==0 && i+1 <argc) {
        if (strcasecmp(argv[i+1],"sift")==0) data_type = DATA_TYPE_SIFT;
        else if (strcasecmp(argv[i+1],"mnist")==0) data_type = DATA_TYPE_MNIST;
        else { fprintf(stderr,"Unknown dataset type\n"); exit(1); }
        i++;
    }
    else if (strcmp(argv[i],"-range")==0 && i+1<argc) {
        if (strcasecmp(argv[i+1],"true")==0) use_range = 1;
        else if (strcasecmp(argv[i+1],"false")==0) use_range = 0;
        else { fprintf(stderr,"Unknown range value\n"); exit(1); 
        } 
        i++;
    }
    else if (strcmp(argv[i],"-hypercube")==0)
        use_hypercube = 1;
    else {
        fprintf(stderr,"Unknown or incomplete option: %s\n", argv[i]);
        fprintf(stderr,"Usage: %s -d <input> -q <query> -kproj <int> -w <double> -M <int> "
                        "-probes <int> -o <output> -N <int> -R <radius>  "
                        "-type <sift|mnist> -range <true|false> -hypercube\n", argv[0]);
        exit(1);
    }
}
    srand(SEED); //Παιρνάμε το φύτρο των ψευδοτυχαιών 
    double *train_dataset=NULL, *query_dataset=NULL;
    int num_train=0, num_query=0, dim=0, query_dim=0; //Αρχικοποιούμε τις διαστάσεις σε 0
    if(data_type==DATA_TYPE_MNIST){ // Φόρτωση datasets (MNIST ή SIFT)
        if(load_mnist_images(train_file_path,&train_dataset,&num_train,&dim)!=0){fprintf(stderr,"Failed loading train\n"); exit(1);}
        if(load_mnist_images(query_file_path,&query_dataset,&num_query,&query_dim)!=0){fprintf(stderr,"Failed loading query\n"); exit(1);}
        for(int i=0;i<num_train*dim;i++) train_dataset[i]/=255.0;
        for(int i=0;i<num_query*dim;i++) query_dataset[i]/=255.0; // Διαιρούμε με το 255 ώστε οι τιμές των pixels να κανονικοποιηθούν στο εύρος [0,1]
    } else {// Φόρτωση SIFT και κανονικοποίηση ώστε να έχουν όλα τα διανύσματα μήκος 1, βοηθά στην απόδοση του αλγορίθμου γιατί συγκρίνουμε μόνο την κατευθυνση κα ιόχι αλλά γνωρίσματα όπωςτο μήκος
        if(load_sift_vectors(train_file_path,&train_dataset,&num_train,&dim)!=0){fprintf(stderr,"Failed loading train\n"); exit(1);}
        if(load_sift_vectors(query_file_path,&query_dataset,&num_query,&query_dim)!=0){fprintf(stderr,"Failed loading query\n"); exit(1);}
        for(int i=0;i<num_train;i++){double norm=0;for(int j=0;j<dim;j++) norm+=train_dataset[i*dim+j]*train_dataset[i*dim+j];norm=sqrt(norm);if(norm>1e-9)for(int j=0;j<dim;j++) train_dataset[i*dim+j]/=norm;}
        for(int i=0;i<num_query;i++){double norm=0;for(int j=0;j<dim;j++) norm+=query_dataset[i*dim+j]*query_dataset[i*dim+j];norm=sqrt(norm);if(norm>1e-9)for(int j=0;j<dim;j++) query_dataset[i*dim+j]/=norm;}
    }
    if(dim!=query_dim){fprintf(stderr,"Tdimensions are wrong\n"); exit(1);}
    hypercube hc;
    if(use_hypercube) initHypercube(&hc,dim,K_PROJ,W); // Αρχικοποίηση της δομής με τις παραμέτρους: dim, K_PROJ, W
    if(use_hypercube){
        for(int i=0;i<num_train;i++){
            insertHypercubeVector(&hc,&train_dataset[i*dim],i,dim); // Εισαγωγή των training διανυσμάτων στη δομή.
        }
    }
    FILE *out = fopen(output_file,"w"); // Άνοιγμα αρχείου εξόδου για εγγραφή
    if(!out){
        perror("Cannot open output file"); 
        exit(1);} 
    double total_af=0.0, total_recall_hits=0.0, total_hc_time=0.0, total_brute_time=0.0; //Ορίζουμε μεταβλητές για τον υπολογισμό των τελικών μέσων όρων
    int max_queries = (num_query>100)?100:num_query; // Θέτουμε σε 100 τον αριθμό των διναυσμάτων που θα εξετάσουμε
    for(int q=0;q<max_queries;q++){ // Για max_queries φορές, θα κάνουμε σύγκρισεις
        const double *query_vec=&query_dataset[q*dim]; //  αντλούμε το query από το trainset
        NearestNeighbor *hc_nn=NULL,*true_nn=NULL; // Ορίζουμε τους ptrs σε ΝΝ για την αποθήκευση των αποτελσμάτων
        int num_R=0,*R_indices=NULL;
        clock_t t_start = clock(); //Αρχιζουμε την χρονομέτρηση
        if(use_hypercube){ hc_nn = find_nn_hc(&hc,query_vec,train_dataset,num_train,N_NEIGHBORS,MAX_CANDIDATES,dim,PROBES); // Εκτελούμε τον αλγόριθμο εύρεσης nn με hc
        } 
        clock_t t_end = clock();
        double hc_time = (double)(t_end-t_start)/CLOCKS_PER_SEC; // Αποθηκεύουμε τον χρόνο πoυ απαιτήθηκε και τον προσθέτουμε στον συνολικό χρόνο ( για τα 100 queries στο τέλος). 
        total_hc_time+=hc_time; 
        clock_t t_start_b=clock(); // Κάνουμε την ίδια διαδικασία για το brute 
        true_nn = BruteForceSearch(train_dataset,num_train,dim,query_vec,N_NEIGHBORS,R_RADIUS,&num_R,&R_indices);
        clock_t t_end_b=clock();
        double brute_time = (double)(t_end_b-t_start_b)/CLOCKS_PER_SEC;
        total_brute_time+=brute_time;
        double query_af_sum=0.0; // Άθροισμα των AF για το τρέχον query
        int valid_af_neighbors=0;// Μετρητής έγκυρων γειτόνων για υπολογισμό AF
        for(int i=0;i<N_NEIGHBORS;i++){
            if(true_nn[i].distance>1e-9 && hc_nn[i].distance<DBL_MAX){ // Υπολογισμός AF: Approximation Factor = D_Approx / D_True.
                query_af_sum+=hc_nn[i].distance/true_nn[i].distance;
                valid_af_neighbors++;
            }
        }
        if(valid_af_neighbors>0){
            total_af+=query_af_sum/valid_af_neighbors; // Προσθήκη του μέσου AF του query στον συνολικό.
        }
        double hits=calculate_recall(hc_nn,true_nn,N_NEIGHBORS);// Υπολογίζουμε πόσους  nn βρήκε και το περνάμε στο συνολικό recall
        total_recall_hits+=hits; 

        fprintf(out,"%s\n", use_hypercube?"Hypercube":"LSH"); // Γράφουμε το header και τα ευρήματα μας για κάθε query στο αρχείο 
        fprintf(out,"Query: %d\n",q);
        for(int i=0;i<N_NEIGHBORS;i++){
            fprintf(out,"Nearest neighbor-%d: image_id_%d\n",i+1,hc_nn[i].index);
            fprintf(out,"distanceApproximate: %.6f\n",hc_nn[i].distance);
            fprintf(out,"distanceTrue: %.6f\n",true_nn[i].distance);
        }
        if(use_range){
            fprintf(out,"R-near neighbors:\n");
            for(int r=0;r<num_R;r++){
                fprintf(out,"image_id_%d\n",R_indices[r]);
            }
        }
        fprintf(out,"\n"); // Απελευθερώνουμε την μνήμη αφού κάναμε τις εγγραφες
        if(hc_nn){free(hc_nn);}
        if(true_nn) {free(true_nn);}
        if(R_indices) {free(R_indices);}
    }
    double avg_af=total_af/max_queries; // Υππολογίζουμε τους μέσους όρους των αποτελεσμάτων για όλα τα queries
    double avg_recall=total_recall_hits/(max_queries*N_NEIGHBORS);
    double avg_hc_time=total_hc_time/max_queries;
    double avg_brute_time=total_brute_time/max_queries;
    double qps=1.0/avg_hc_time;
    fprintf(out,"Average AF: %.4f\n",avg_af); // Tους περνάμε στο αρχείο
    fprintf(out,"Recall@%d: %.4f\n",N_NEIGHBORS,avg_recall);
    fprintf(out,"QPS: %.2f\n",qps);
    fprintf(out,"tApproximateAverage: %.6f\n",avg_hc_time);
    fprintf(out,"tTrueAverage: %.6f\n",avg_brute_time);
    fclose(out);// τελειώνουμε τις εγγραφές και κελέινουμε το αρχείο
    if(use_hypercube){// Αποδεσμέυουμε την μνήμη και επιστρέφουμε μύνημα επιτυχίας
        FreeHypercube(&hc);
    }  
    free(train_dataset);
    free(query_dataset);
    printf("Success!");
    return 0;
}