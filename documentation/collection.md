# Phrase Collection

@(English)



### Biology Terminology

- underlying genetic architecture
- genome-wide association studies and linkage mapping / assaying genetic variations
- (Y data) linical phenotypes or disease susceptibility / phenotype variability / complex traits
- (X data) single nucleotide polymorphisms / individual loci / common SNP-based variants
- (linkage mapping) incorporate such information on relatedness of genes into statistical analysis of associations between SNPs and gene expressions / multifactorial associations / joint identification of sets of loci / 
- Constrained least squares problem

### Our methods

- Joint modeling of population structure and individual SNP effects
- The rigorous combination of Lasso and mixed modeling approaches
- Our approach is simple and free of tuning parameters, effectively controls for population structure and scales to genome-wide datasets.

### Successful Application

- mapping yielded insights into the genetic architecture of global-level traits in plants (Atwell et al., 2010) and mouse (Valdar et al., 2006), as well as the risks for important human diseases such as type 2 diabetes (Craddock et al., 2010)

### Challenge

- single genetic variants rarely explain larger fractions of phenotype variability, and hence, individual effect sizes are small(McCarthy et al., 2008; Mackay et al., 2009)
- Population structure, family structure and cryptic relatedness can induce false association patterns with large numbers of loci being correlated with the phenotype.
- A major source of these effects can be understood as deviation from the idealized assumption that the samples in the study population are unrelated. Instead, population structure in the sample is difficult to avoid and even in a seemingly stratified sample, the extent of hidden structure cannot be ignored (Newman et al., 2001).

### Previous method

- examine the expression level of a single gene at a time for association, treating genes as in- dependent of each other [Cheung et al. (2005), Stranger et al. (2005), Zhu et al. (2008)]
- However, it is widely believed that many of the genes in the same biolog- ical pathway are often co-expressed or co-regulated [Pujana et al. (2007), Zhang and Horvath (2005)]
- Applying a Laplace prior leads to the Lasso (Li et al., 2011), and related priors have also been considered (Hoggart et al., 2008)
- With the same ultimate goal to capture the genetic effects of groups of SNPs, variance component models have recently been proposed to quantify the heritable component of phenotype variation explainable by an excess of weak effects (Yang et al., 2010).

__Population Structures__

- EIGENSTRAT builds on the idea of extracting the major axes of population differentiation using a PCA decomposition of the genotype data (Price et al., 2006), and subsequently including them into the model as additional covariates
- Linear mixed models (Yu et al., 2006; Kang et al., 2008; Zhang et al., 2010; Kang et al., 2010; Lippert et al., 2011) provide for more fine-grained control by modeling the contribution of population structure as a random effect, providing for an effective correction of family structure and cryptic relatedness.

__Jointly__

- Hoggart et al. (2008), Li et al. (2011) add principal components to the model to correct for population structure. 
- In parallel to our work, Segura et al. (2012) have proposed a related multi-locus mixed model approach, however employing step-wise forward selection instead of using the Lasso.

## 1. Sentence

1. Recent advances in ... have provided researchers ... opportunity to comprehensively study the genetic causes of complex diseases

2. The past year has witnessed / seen a remarkable shift in our capacity to do

## 2. Adjective

### Good

- Unprecedented / miraculous
- remarkable / top-notched / first-class
- conceptually simple, computationally efficient and scales to genome-wide settings
- Sophisticated / fine-grained
- well-suited
- prove itself extremely well-suited to sth

### Bad

- fall short
- impractical

### Impossible

- rule out

### Hard

- complicated

### Many

- substantial
- 

### Unsolved

- under-addressed

## 3. Verb

### Can do sth

- Address the challenge
- Propose a new approach
- Allow (the database that allows the process of balabala)

### Like

- In parallel to

### Access

- evaluate / access / 

### Combime

- bridge sth with sth

### Make / help

- give / provide sb an ... opportunity
- provide a deeper insight into ... by ...
- yields greater power to do sth

### Use

- employ

### Guess

- speculate

### be Interested in 

- be hooked on / 

## 4. Norm

### Way

- method / manner / fashion / avenue

## 5. Logic

### Causality

- give rise to / induce
- in light of 
