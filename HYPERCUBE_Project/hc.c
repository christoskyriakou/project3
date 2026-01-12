#include "hc.h"
#include <time.h>
int f_funct(long long h_val) {
    h_val = h_val * 1103515245; 
    return (int)(llabs(h_val) & 1);
}
long long h_funct(const double* vector, const double* r, double t, int dim, double W){// Η h_funct, υπολογίζει τα h τα οποία θα χρησιμοποιήσει η get_g για να βγάλουμε το τελικό g και να βρούμε σε ποιό bucket θα καταλήξει το διάνυσμα
    double scalar_prod = scalar_product(vector, r, dim); // Αρχικά παίρνουμε το εσωτερικό γινόμενο του διανύσματος με το τυχαίο διάνυσμα Ρ
    double result = (scalar_prod + t) / W; // Ύστερα επιστρέφουμε το κάτων όριο του εσωτερικού γινομένου με μία μετατόπηση t διαιρεμένο από το μήκος κελιού. Εφαρμόζουμε λοιπόν τον τύπο των διαφανειών.
    return (long long)floor(result); // Επιστρέφουμε το h σε μορφή long long Καθώσς στο sift είχαμε προβληματα με overflow και segmentation fault
}

int get_g(hypercube* hc,const double* vector,int dim){
    int g=0; // Αρχικοποιούμε τον δείκτη
    for(int i=0;i<hc->kproj;i++){//Επαναλαμβάνουμε k_proj φορές για να κατασκευάσουμε έναν δυαδικό δείκτη k_proj-bit
        long long h_val =h_funct(vector,hc->r[i],hc->t[i],dim, hc->W);  // Υπολογίζπθμε το h, καλουμε την f funct η οποία μας επιστρέφει τον αριθμό 0 η 1 αφού μπερδέψει τον αριθμό 
        int bit = f_funct(h_val); 
        g=(g<<1)|bit;  // Μετακινούμε όλα τα υπάρχοντα bits του g μία θέση αριστερά και προσθέτουμε το νέο bit
    }
    return g; // Επιστρέφει τον τελικό ακέραιο δείκτη 
}

double randn(){
    double sum=0.0;
    // Χρησιμοποιούμε το Κεντρικό Οριακό Θεώρημα για προσέγγιση N(0, 1)
    for(int i=0;i<12;i++){
        sum +=(double)rand()/RAND_MAX; // Uniform(0, 1)
    }
    return sum-6.0; // Προσεγγίζει N(0, 1)
}
void initBucket(bucket *b){
    b->count=0; // μετρητής του αριθμού των αποθηκευμένων διανυσμάτων στο εκάστοτε μπάκετ, αρχικά 0
    b->capacity=100; // Ορίζουμε το capacity, την χωριτητικότητα του πίνακα vectors στον οποίο θα αποθηκευτούν τα διανύσματα του bucket
    b->vectors=(int*)(malloc(sizeof(int)*(b->capacity))); // Κάνουμε type casting, διότι η malloc δεν θέλουμε να επιστρέψει απλά ένα *void αλλά ένα pointer σε int
    if(b->vectors==NULL){ // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα
        perror("ERROR in initialization of Bucket");
        exit(1);
    }
}
void resize_bucket(bucket* b){ // Συνάρτηση για επεκταση μνήμης του πίνακα vectors
    int new_capacity=b->capacity*2; // Ορίζουμε την νέα χωρητικότητα σε διπλάσια
    int* temp_ptr=(int*)realloc(b->vectors,new_capacity* sizeof(int)); // Κάνουμε realloc σε έναν αρχικά προσωρινό πίνακα, με το νέο capacity
    if (temp_ptr==NULL){ // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα 
        perror("ERROR in initialization of Bucket");
        exit(1);    
   }
    b->vectors=temp_ptr; //θέτουμε τον πινακα vectors να ειναι ο ανανεωμένος πίνακας και επικαιροποιούμε και το capacity 
    b->capacity=new_capacity;
}
void addToBucket(bucket *b,int vector){ // Σύναρτηση για την εισαγωγή ενός vector στο bucket
    if(b->count>=b->capacity){ // Αν ο αριθμός των διανυσμάτων ειναι ίσως με την χωρητικότητα, το bucket ειναι γεμάτο και καλούμε την resize
    resize_bucket(b);
    }
    b->vectors[b->count]=vector; // Σε άλλη περίπτωση, βάζουμε στο τέλος του πίνακα vectors το νέο μας διάνυσμα 
    b->count= (b->count)+1;// Αυξάνουμε το count κατά 1 και έτσι στην επόμενη εισαγωγή, το διάνυσμα μπαίενι στην επόμενη άδεια θέση
}

void freeBucket(bucket* b){ // Η συνάρτηση κάνει την αντίθεση διαδικασία από την initialize, απελευθερώνει την δεσμευμένη δυναμικά μνήμη και θέτει τους ptrs σε NULL τιμή.
    if(b==NULL){ // Αν το b είναι NULL, τότε έχει γίνει free το bucket
        return;
    }
    if(b->vectors!=NULL){ //Αν σεν ειναι null ο δείκτης στον πίνακα των διανυσμάρων, καλούμε την free για τον πίνακα και μετά κάνουμε null τον ptr στον πίνακα
        free(b->vectors);
        b->vectors=NULL;
    }
    b->count=0; // θέτουμε σε 0 τις τιμές των μετρητών.
    b->capacity=0;
}

void insertHypercubeVector(hypercube* hc,const double* vector_v,int vector_index,int dim){
    int g =get_g(hc,vector_v,dim);    // Υπολογίζουμε το g για το διάνυσμα
    addToBucket(&(hc->buckets[g]),vector_index);  // Προσθέτουμε τον δείκτη του διανύσματος στον αντίστοιχο κάδο
}

NearestNeighbor* find_nn_hc(hypercube* h,const double* q,const double* dataset,int num_images,int N,int max_candidates_M,int dim,int probes) {
    int initial_vertex_index=get_g(h,q,dim);  // Υπολογίζουμε το g για το q 
    int num_vertices=1<<h->kproj;  // Υπολογίζουμε πόσοι κόμβοι υπάρχουν συνολικά 
    NearestNeighbor* candidates_list=(NearestNeighbor*)malloc(sizeof(NearestNeighbor)*max_candidates_M);  // δεσμεύουμε δυναμικά πίνακα με δεικτεε σε nn
    if(candidates_list==NULL){
        perror("error in find nn hc");
        exit(1);
    }
    int number_of_candidates=0; 
    int *visited_vertices=(int*)calloc(num_vertices,sizeof(int));  // δημιουργούμε έναν άδειο πίνακα που θα αποθηκεύει τα διανύσματα που έχουμε βρει
    int *is_added=(int*)calloc(num_images,sizeof(int));    // Δεσμεύουμε χώρο για έναν πίνακα ptr σε int για να εξασφαλιστεί ότι κάθε διάνυσμα εισάγεται μόνο μία φορά στη λίστα υποψηφίων
    int *queue=(int*)malloc(sizeof(int)*num_vertices); // Ουρά για την BFS
    int head = 0, tail = 0; // Δείκτες αρχής και τέλους της ουράς
    if(!candidates_list||!visited_vertices||!is_added||!queue){  // Έλεγχος αποτυχίας δέσμευσης μνήμης
        free(candidates_list); 
        free(visited_vertices); 
        free(is_added); 
        free(queue);
        return NULL;
    }
    queue[tail++] = initial_vertex_index;    // Ξεκινάμε την BFS από τον κόμβο που αντιστοιχεί στο ερώτημα q και το αποθηκεύουμε στο visited 
    visited_vertices[initial_vertex_index] = 1;
    int vertices_examined=0;// Αρχικοποίηση μετρητή διανυσμάτων που έχουμε ελέγξει 
    while(head!=tail&& vertices_examined<probes)  // όσο δεν έχουμε διατρέξει όλη την ουρά και δεν έχουμε εξετάσει περισσότερα διανύσματα από το P συνεχίζουμε την αναζήτηση
    {
        if(number_of_candidates>=max_candidates_M){ // Αν ο αριθμός των υποψηφίων ειναι μεγαλύτερος από αυτόν του μέγιστου αρ υποψηφιών κάνουμε break
            break; 
        }
        int current_vertex_index=queue[head++];   // Παίρνουμε τον επόμενο vertex από την ουρά BFS
        bucket* current_bucket=&h->buckets[current_vertex_index]; // Αντιστοιχούμε το vertex στο αντίστοιχο bucket
        vertices_examined++; // Αυξάνουμε τον μετρητή εξετασμένων vertices
        for (int i=0;i<current_bucket->count;i++) {  // Για κάθε vector στο bucket
            if (number_of_candidates>=max_candidates_M) {
                break; // Όριο M (υποψηφίων) επιτεύχθηκε
            }
            int vector_index = current_bucket->vectors[i]; // Παίρνουμε το index του vector
            if(is_added[vector_index]==0){//  Αν το vector δεν έχει ήδη προστεθεί
                const double *current_vector=&dataset[vector_index*dim];  // Δείκτης στο vector
                double distance=euclidean_distance(q, current_vector,dim);// Υπολογισμός απόστασης
                candidates_list[number_of_candidates].index=vector_index;// Αποθηκεύουμε το index kai την απόσταση
                candidates_list[number_of_candidates].distance=distance;
                number_of_candidates++; // αυξάνουμε τον αριθμό υποψηφίων και σημειώνουμε ότι προστέθηκε
                is_added[vector_index]=1; 
            }
        }
        for(int i=0;i<h->kproj;i++){  // Για κάθε bit του vertex, υπολογίζουμε το index του γειτονικού και αν δεν το έχουμε επισκφτεί, το επισκεφτόμστε και το βάζουμε στην ουρά
            int neighbor_index=current_vertex_index^(1<<i); 
            if (visited_vertices[neighbor_index]==0) {
                visited_vertices[neighbor_index]=1;
                queue[tail++]=neighbor_index;
            }
        }
    }
    qsort(candidates_list, number_of_candidates,sizeof(NearestNeighbor),compareNeighbors); // Καλούμε την qsort μέ την compareNeighbor και έτσι σορτάρουμε τους υποψήφιους γειτονες
    NearestNeighbor* K_NNs =(NearestNeighbor*)malloc(sizeof(NearestNeighbor)*N);// δεσμευση μνήμης για το αποτέλεσμα των Ν γειτόνων
    for(int i=0;i<N;i++){
        if(i<number_of_candidates){ // Παίρνουμε μονάχα τους Ν γείτονες αν ειναι λιγότεροι συνεχίζουμε να γεμίζουμε το result με κενές θέσεις.
            K_NNs[i]=candidates_list[i];
        }else{
            K_NNs[i].distance=DBL_MAX; 
            K_NNs[i].index=-1;
        }
    }
    free(candidates_list); // Καθαρίζουμε την μνήμη και επσιτρέφουμε το αποτέλεσμα
    free(visited_vertices);
    free(is_added);
    free(queue);
    return K_NNs;
}

void initHypercube(hypercube* h,int dim,int k,double W){ // Η συνάρτηση αυτή αρχικοποιεί τη δομή του hypercube
    h->kproj=k; // Αντλούμε τον αριθμό των διαστασεων Κ και ρο μήκος κελιού 
    h->W=W;
    int bucket_size = (1 << k); // Το bucket size ειναι 2^k σύμφωνα με τις αρχές του hypercube
    h->buckets=(bucket*)malloc(sizeof(bucket)*bucket_size); // Δεσμεύουμε δυναμικά μνήμη για τον πίνακα των buckets.
    if(h->buckets==NULL){
        perror("ERROR! in Hypercube initialization"); 
        exit(1);
    }
    for(int z=0;z<bucket_size;z++){  // Kάνουμε initialize τα επιμέρους buckets
        initBucket(&(h->buckets[z]));
    }
    h->r=(double**)malloc(sizeof(double*)*h->kproj);// Δεσμέυουμε πίνακα ώστε να αποθηκεύσουμετα r διανύσματα
    if(h->r==NULL){
        perror("ERROR! in Hypercube initialization"); 
        exit(1);
    }
    h->t=(double*)malloc(sizeof(double)*h->kproj); // Δεσμέυουμε χώρο για τα t
    if (h->t==NULL){
        perror("ERROR! in Hypercube initialization");
        exit(1);
    }
    for(int i=0;i<h->kproj;i++){ // Δεσμεύουμε χώρο ώστε τα διανύσματα r να τα αποθηκεύσουμε
        h->r[i]=(double*)malloc(sizeof(double)*dim);
        if(h->r[i]==NULL){
            perror("ERROR! in Hypercube initialization");
            exit(1);
        }
    }
    for(int i=0;i<h->kproj;i++){
        h->t[i]=(double)rand()/(double)RAND_MAX*h->W; // Για κάθε t θέτουμε τυχαία τιμή και για τα kproj διανυσματα r μεγέθους dim 
        for(int j=0;j<dim;j++){
            h->r[i][j]=(double)randn();
        }
    }
}

void FreeHypercube(hypercube* h){
    if(h==NULL){ //Αν το h είναι NULL, τότε έχει ολοκληρωθεί η απελευθέρωση
        return;
    }
    int bucket_size=(1<<h->kproj); // Υπολογίζουμε το πλήθος των buckets του hypercube (2^kproj).
    if(h->buckets!=NULL){        // Απελευθέρωση όλων των 2^k bucket
        for(int i=0;i<bucket_size;i++){ 
            freeBucket(&h->buckets[i]);// Καλούμε την freebucket και θέτουμε τους αντιστοιχους ptr σε null
        }
        free(h->buckets);
        h->buckets=NULL;
    }
    if(h->r!=NULL){// Ελέγχουμε αν έχουν δεσμευθεί τα τυχαία διανύσματα προβολής r.
        for(int j=0;j<h->kproj;j++){ // Ελευθερώνουμε κάθε επιμέρους διάνυσμα προβολής.
            if(h->r[j]!=NULL){
                free(h->r[j]);
                h->r[j]=NULL;
            }
        }
        free(h->r);
        h->r=NULL;
    }
    if(h->t!=NULL){// Απελευθερώνουμε τον πίνακα αν δεν έχει απελευθερεωθει
        free(h->t);
        h->t=NULL;
    }
}

int hamming_dist(int a,int b){ // Υπολογίζουμε την απόσταση Hamming μεταξύ δύο ακεραίων a και b 
    unsigned int xor_result=a^b; // κάνουμε την λογική πράξη xor και με το counter μετράμε τους άσους
    int count=0;
    while(xor_result!=0){ // όσο δεν έχουμε διατρέξει όλον τον αριθμό κάνουμε sift και παίρνουμε το lsb και ελεγχουμε αν ειναι 1
        if(xor_result & 1){
            count+=1;
        }
        xor_result>>=1;
    }
    return count;// Επιστρέφουμε τον αριθμό διαφορετικών bits
}

double euclidean_distance(const double *v1,const double *v2,int dim){  // Υπολογίζουμε την ευλκείδια απόσταση με βάση τον τύπο 
    double sum=0.0;
    for(int i=0;i<dim;i++){ 
        double dif=v1[i]-v2[i]; // Άθριζουμε τα τετραγωνα των διαφορών σε κάθε διάσταση
        sum+=dif*dif;
    }
    return sqrt(sum); // Επιστρέφουμε την αποσταση
}
double scalar_product(const double *vector_v,const double *vector_r,int dim){ // Η συνάρτηση αυτή παίρνει ως όρισμα δύο ptrs σε διανύσματα double, και την διάσταση των διανυσμάτων ( ώστε αν μπορεί να χρησιμοποιηθεί τόσο από τα διανύσματα του mnsit όσ και του sift database)
    double result=0.0; // αρχικοποίησ του αποτελέσματος που θα είναι το εσωτερικό γινόμενο των διανυσμάτων
    for(int i=0;i<dim;i++){ // Με ένα loop, παίρνουμε καθένα από τα μέρη του διανύσματος μέχρι το dim ( το πέρας του)
        result+= vector_v[i]*vector_r[i]; // Εκτελούμε των πολλαπλασιμό τους και το προσθέτουμε στο result 
    }
    return result; // επιστρέφεται το εσ. γινόμενο 
}

int compareNeighbors(const void* a,const void* b){ // παιρνει δύο γενικά ορισματα ptr σε void
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
