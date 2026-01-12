#include "lsh.h" // Περιολαμβάνουμε τα structs και τις δηλώσεις των πρωτότυπων συναρτήσεων από το lsh.h, μέσω της αντίστοιχης κεφαλίδας
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

double scalar_product(const double *vector_v,const double *vector_r,int dim){ // Η συνάρτηση αυτή παίρνει ως όρισμα δύο ptrs σε διανύσματα double, και την διάσταση των διανυσμάτων ( ώστε αν μπορεί να χρησιμοποιηθεί τόσο από τα διανύσματα του mnsit όσ και του sift database)
    double result=0.0; // αρχικοποίησ του αποτελέσματος που θα είναι το εσωτερικό γινόμενο των διανυσμάτων
    for(int i=0;i<dim;i++){ // Με ένα loop, παίρνουμε καθένα από τα μέρη του διανύσματος μέχρι το dim ( το πέρας του)
        result+= vector_v[i]*vector_r[i]; // Εκτελούμε των πολλαπλασιμό τους και το προσθέτουμε στο result 
    }
    return result; // επιστρέφεται το εσ. γινόμενο 
}
double euclidean_distance(const double *v1,const double *v2,int dim){  // Υπολογίζουμε την ευλκείδια απόσταση με βάση τον τύπο 
    double sum=0.0;
    for(int i=0;i<dim;i++){ 
        double dif=v1[i]-v2[i]; // Άθριζουμε τα τετραγωνα των διαφορών σε κάθε διάσταση
        sum+=dif*dif;
    }
    return sqrt(sum); // Επιστρέφουμε την αποσταση
}
int h_funct(const double* vector_v,const double* vector_r,double t,int dim){  // Η h_funct, υπολογίζει τα h τα οποία θα χρησιμοποιήσει η get_g για να βγάλουμε το τελικό g και να βρούμε σε ποιό bucket θα καταλήξει το διάνυσμα
    double dot_vector=scalar_product(vector_v,vector_r,dim); // Αρχικά παίρνουμε το εσωτερικό γινόμενο του διανύσματος με το τυχαίο διάνυσμα Ρ
    return((int)floor((dot_vector + t)/W)); // Ύστερα επιστρέφουμε το κάτων όριο του εσωτερικού γινομένου με μία μετατόπηση t διαιρεμένο από το μήκος κελιού. Εφαρμόζουμε λοιπόν τον τύπο των διαφανειών.
} 

int get_g(const double *vector_v,hashtable* h,int dim){// Υπολογισμός του g, κάθε g αποτελείται από Κ h συναρ΄τησεις κατακερματισμού
    long long g=0; // θέτουμε την long long g προς αποφυγή του overflow 
    long long M_long=(long long)M_HASH_SIZE; // αντίστοιχα μέσω του type cast ορίζουμε την M_long ώστε να αποφύγουμε τοoverflow
    for(int i=0;i<K;i++){// Για Κ φορές, 
        long long h_val_temp=(long long)h_funct(vector_v,h->r[i],h->t[i],dim); // υπολογίζουμε την long long h val, μέσω της h_funct
        long long h_val=(h_val_temp % M_long + M_long)%M_long; // Το (x%M+M)%M εξασφαλίζει ότι το αποτέλεσμα είναι πάντα θετικό, στο εύρος [0,M-1]
        long long v_val=h->v[i]; // Αντλούμε την τιμή του v για το συγκεκριμενο i από το hashtable
        long long term=(v_val*h_val)%M_long; //Υπολογίζουμε τον όρο (v_i *h_i) mod M
        g=(g+term)%M_long; // Προσθέτουμε το term στο συνολικό g, κρατώντας το modulo M
    }
    return(int)g; // Επιστρέφουμε το g σε int
}
void insertVector(hashtable* h,const double* vector_v,int vector_index,int dim){ // Συναρτηση εισαγωγής ενός διανύσνματος σε bucket
    int bucket_g= get_g(vector_v,h,dim); // Arxikα καλούμε την get_g και παίρνουμε το g του bucket
    addToBucket(&(h->buckets[bucket_g]),vector_index); // Βάζουμε το vector_v στο σωστό bucket, χρησμιποιούμε & καθώς θέλουμε να κάνουμε αλλαγε΄ςσ την δομή και έτσι χρειαζόμαστε την διευθυνση στην μνήμη
}

void initHashTable(hashtable* h,int dim){ // Παίρνει ως όρισμα έναν δείκτη σε hashtable και την διάσταση dim των διανυσμάτων.
    h->buckets=(bucket*)malloc(sizeof(bucket)*M_HASH_SIZE); // Δεσμεύουμε δυναμικά χώρο στην μνήμη όπου θα αποθηκεύσουμε τα buckets του κάθε hashtable 
    if(h->buckets==NULL){ // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα
        perror("ERROR in initialization of Hashtable");
        exit(1);
    }  
    for(int i=0;i<M_HASH_SIZE;i++){ // Με ένα loop πήγαίνουμε στο κάθε bucket, μέχρι το M_HASH_SIZE και καλούμε την initBucket
        initBucket(&(h->buckets[i])); // Χρησιμοποιούμε & καθώς θέλουμε να κάνουμε αλλαγές στην δομή και πρέπει να έχουμε την διευθυνση της μνήμης.
    }
    
    h->r=(double**)malloc(sizeof(double*)*K);  // Δεσμεύουμε δυναμικά μνήμη για έναν ptr σε ένα πίνακα δεικτών σε double, όπου θα ειναι τα διανσύματα r. θα έχει μέγεθος K, όσες και οι φορές που γίνεται η σειρά Σ του υπολογισμού του g
    if(h->r==NULL){ // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα
        perror("ERROR in initialization of Hashtable");
        exit(1);
    }  
    for(int i=0;i<K;i++){// Για τα Κ διανύσματα θα πρέπει να γίνει η δυναμική δέσμευση του χώρου τους,
        h->r[i]=(double*)malloc(sizeof(double)*dim); // Δεσμεύουμε πάλι χώρο αυτη την φορά ptr σε double , αφού θ απαοθηκεύσουμε τα ίδια τα διανύσματα μήκους πάντα dim
        if(h->r[i]==NULL){ // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα
            perror("ERROR in initialization of Hashtable");
            exit(1);
        }
    }
    for(int i=0;i<K;i++){ // Αρχικοποιούμε τα Κ διανύσματα
        h->t[i]=(double)(rand()/(double)RAND_MAX)*W; // Τώρα, θα υπολογίσουμε το μέγεθος της τυχαίας μετατόπισης t, η οποία θα πρέπει  να είναι πολλαπλάσια του μήκους των κελιών w
        double norm=0.0;         // Αρχικοποιούμε το norm για να υπολογίσουμε το μήκος του τυχαίου διανύσματος r_i
        for(int j=0;j<dim;j++){ // Για το κάθε i διάνυσμα μήκους j, αρχικοπούμε την 
            h->r[i][j]=randn();  // Παίρνουμε τυχαίο αριθμό από κανονική κατανομή
            norm+=h->r[i][j]*h->r[i][j]; // Προσθέτουμε το τετράγωνο της συνιστώσας στο norm 
        }
        norm=sqrt(norm); // Τώρα norm είναι η Ευκλείδεια νόρμα 
        if(norm >1e-10){  // Αποφυγή διαίρεσης με 0 , βαζουμε ένα πολύ μικρό θετικό 
            for(int j=0;j<dim;j++){ // Ελέγχουμε ότι το μήκος δεν είναι σχεδόν μηδέν
                h->r[i][j]/=norm;// Διαιρούμε κάθε στοιχείο με το συνολικό μήκος του διανύσματος
            }
        }
        h->v[i]=(rand()%100000)+1;// Επιλέγουμε τυχαία τιμή για το v i (χρησιμοποιείται στον τελικό υπολογισμό g) η οποία  παίρνει τιμές στο εύρος 1 έως 100000
    }
}
void freeHashTable(hashtable* h){ // Απελευθερώνει την δεσμευμένη δυναμικά μνήμη για το hashtable και θέτει τους ptrs σε NULL τιμή.
    if(h==NULL){ //Αν το h είναι NULL, τότε έχει ολοκληρωθεί η απελευθέρωση
        return;
    }
    if(h->buckets!=NULL){ // Αν ο πίνακας με τα buckets δεν ειναι NULL, κάνουμε freebucket Μ φορές για τα buckets και στην συνέχει αποδεσμεύουμε κατάλληλα τον πίνακα των buckets
        for(int i=0;i<M_HASH_SIZE;i++){
            freeBucket(&(h->buckets[i]));
        }
        free(h->buckets);
        h->buckets=NULL;
    }
    if(h->r!=NULL){ // Aν δεν είναι NULL ο πινακας με τα διανύσματα r πάμε να τον αποδεσμεύσουμε
        for(int i=0;i<K;i++){ // Μέσω λουπ, για Κ διανύσματα, αν δεν είναι NULL το διάνυσμα καλούμε free
            if(h->r[i]!=NULL){
            free(h->r[i]);
            }
        }
        free(h->r); // κάνουμε free για τον πίνακα των r και 
        h->r=NULL;// και θέτουμε NULL τον ptr του hash table στο r
    }
}

NearestNeighbor* find_l_nn(hashtable lsh_tables[], const double* query_vec, const double* train_dataset, int num_train_vectors, int N, int dim){
    NearestNeighbor* candidates = malloc(sizeof(NearestNeighbor) *num_train_vectors);  // Δημιουργία πίνακα για αποθήκευση όλων των υποψηφίων γειτόνων
    if(candidates==NULL){ // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα
        perror("ERROR in find l nn 1"); 
        return NULL;
    }
    int* visited=calloc(num_train_vectors,sizeof(int));  // Καλούμε calloc, ώστε να αποθηκεύσουμε μονάχα τον ακριβή αριθμό διανυσμάτων 
    if(visited==NULL){  // Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα
        perror("ERROR in find l nn 2");
        free(candidates); // Ελευθερώνουμε την δεσμευμένη μνήμη των candidates ώστε να μην έχουμε mem leaks
        return NULL;
    }
    int candidate_count=0; // Αρχικοποίηση μετρητή υποψηφίων γειτόνων
    for(int i=0;i<L;i++){ // Για τους L πίνακες hashtables, 
        hashtable* h=&lsh_tables[i]; // Δείκτης στον τρέχον hashtable
        int g_val=get_g(query_vec,h,dim); // Παίρνουμε το g και βάζουμε το b να δείχνει στο αντίστοιχο bucket του g
        bucket* b=&h->buckets[g_val];
        for(int j=0;j<b->count;j++){ // εξετάζουμε όλα τα διανύσματα του β
            int idx=b->vectors[j]; //Αντλούμε το index από το vectors και αν δεν ανοίκει στα visited, το παιρνάμε
            if(!visited[idx]){
                visited[idx]=1;
                const double* train_vec=&train_dataset[idx*dim]; // παίρνουμε το πραγματικό διάνυσμα από το train_data set πολλαπλασιάζοντας με dim και στην συνέχεια υπολογίζουμε την ευκλέιδια απόσταση 
                double dist=euclidean_distance(query_vec,train_vec,dim);
                candidates[candidate_count].index=idx; // παίρνάμε τα στοιχεία του διανύσματος στα υποψήφια διανύσματα ΝΝ
                candidates[candidate_count].distance=dist;
                candidate_count++;
            }
        }
    }
    if(candidate_count==0){  // Δεν βρέθηκαν υποψήφιοι
        NearestNeighbor* no_neighbours=malloc(sizeof(NearestNeighbor)*N); // ορίζουμε την κενή λίστα γειτόνων
        if(no_neighbours==NULL){
            perror("ERORR in fin l nn 4");
            exit(1);
        }
        for(int i=0;i<N;i++){
            no_neighbours[i].index=-1;
            no_neighbours[i].distance=INFINITY;
        }
        free(candidates);
        free(visited);
        return no_neighbours; // 
    }

    
    qsort(candidates,candidate_count,sizeof(NearestNeighbor),compareNeighbors); // Καλούμε την qsort μέ την compareNeighbor και έτσι σορτάρουμε τους υποψήφιους γειτονες
    NearestNeighbor* result=malloc(sizeof(NearestNeighbor)*N); // δεσμευση μνήμης για το αποτέλεσμα των Ν γειτόνων
    if(result==NULL){// Σε περίπτωση που η malloc αποτύχει, επιστρέφεται σφάλμα 
        perror("ERROR in find l nn 3"); 
        free(candidates);
        free(visited);
        return NULL;
    }
    for (int i=0;i<N;i++){ // Παίρνουμε μονάχα τους Ν γείτονες αν ειναι λιγότεροι συνεχίζουμε να γεμίζουμε το result με κενές θέσεις.
        if(i<candidate_count){
            result[i]=candidates[i];
        }else{
            result[i].index=-1;
            result[i].distance=INFINITY;
        }
    }
    free(candidates); // Καθαρίζουμε την μνήμη και επσιτρέφουμε το αποτέλεσμα
    free(visited);
    return result;
}


