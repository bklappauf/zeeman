## -*- coding: iso-8859-1 -*-
"""
This module contains two classes:
The Zeeman Class 
    - main module that can calculate level splittings for j manifold at a fixed magnetic field
The QuantumNumbers Class
    - convenience class for providing preset quantum numbers for different isotopes
use
from zeeman_module import Zeeman,QuantumNumbers
then add plotting code as shown below

"""
import scipy
from scipy.misc import factorial
import numpy
import pylab
from physcon import mu_B,hbar,h,u


class Zeeman(object):
    '''
    Class to calculate Zeeman shift for a given LSIJ state for a given
    alkali atom.
    the program will automatically calculate all associated hyperfine states.
    Certain coefficients must be provided for the given atom.
    Ahfs  in Joules
    Bhfs  in Joules (for j != 1/2)
    (and in principle gs,gl,gi though these don't change by much so these are 
    probably not too critical' see #Rev. Mod. Phys. 49, 31 (1977).)

    note that in the high field regime:
    	The energies are then given to lowest order by 
	E|J mJ I mI>= 
	Ahfs mJ mI + 
	Bhfs*{3(mJ mI )**2 + 3/2 mJ mI − I(I + 1)J(J + 1)/(2J(2J − 1)I(2I − 1))}+ 
	μB (gJ mJ + gI mI )Bz .
	The energy shift in this regime is called the Paschen-Back effect.
    so the even splitting in each m_j manifold dependent on m_i is 
    from the first term here.
    '''
    
    # fields used to find state labels in limit of fine and hyperfine basis regimes
    FINE_STRUCTURE_SEPARATION_FIELD = 1e0  
    HYPERFINE_STRUCTURE_SEPARATION_FIELD = 1e-5 
    
    def __init__(self,L,S,I,J,Ahfs,Bhfs,gi):
        self.LSIJ = (L,S,I,J)
        self.Ahfs = Ahfs
        self.Bhfs = Bhfs
        self.H_to_MHz = 1e-6/h
        self.fstates,self.jstates = self.states(I,J)
        self.N = len(self.fstates)
        self.gi = gi  #-.000293 for 85, .000477 for Li6
                            #Rev. Mod. Phys. 49, 31 (1977).
        self.g_s()
        self.cg_coef = self.C(I,J)
        self.Hb_jbasis = self.Hb_j(self.jstates)
        self.Hb_fbasis = self.Hb_f()
        self.Hhfs_fbasis = self.Hhfs(self.fstates)
        self.energies = self.Hhfs_fbasis.diagonal()
        self.ijlabels = self.getLabels_ij()
        self.fmlabels = self.getLabels_fm()
        

    def delta(self,x):
        # dirac delta function
        if x == 0:
            return 1
        else:
            return 0

    def fac(self,x):
        # factorial which makes neg values 0
        if x>=0:
            return factorial(x)
        else:
            return 0

    def states(self,I,J):
        #defines the vectors of N states for (f,mf) and (mi,mj) basis
        fmax = I+J
        fmin = scipy.absolute(I-J)
        fstates = []
        for f in scipy.arange(fmax,fmin-1,-1):
            for m in scipy.arange(f,-f-1,-1):
                fstates.append((f,m))
        ijstates = []
        for mi in scipy.arange(-I,I+1):
            for mj in scipy.arange(-J,J+1):
        #for mj in scipy.arange(-J,J+1):
            #for mi in scipy.arange(-I,I+1):
                ijstates.append((mi,mj))
        ijstates     #.reverse()
        return fstates,ijstates
    
    def g_s(self):
        # defines the g values ... these in principle could be changed 
        # these are hardwired for Rb87 right now
        L,S,I,J = self.LSIJ
        self.gs = 2.002319304 
        self.gl = .999993
        self.gj = self.gl*( (J*(J+1)-S*(S+1)+L*(L+1))/(2*J*(J+1)) )\
                 +self.gs*( (J*(J+1)+S*(S+1)-L*(L+1))/(2*J*(J+1)) )

    def cg(self,j11,m11,j22,m22,j,m):
        # calculates a clebsch-gordan coef (j1,m1,j2,m2,j,m).
        # uses general cg formula provided in many references.
        # assume qf is index for fstates and qj for ijstates.
        # fstates[qf] = sum_over_qj( C(qf,qj)*(ijstates[qj] )
        fac = self.fac
        if j11 < j22: #if j1<j2 switch them
            j1 = j22
            j2 = j11
            m1 = m22
            m2 = m11
        else:
            j1 = j11
            j2 = j22
            m1 = m11
            m2 = m22
        N1 = fac(j1+j2-j)*fac(j+j1-j2)*fac(j+j2-j1)*(2*j+1)
        D1 = fac(j+j1+j2+1)	
        N2 = fac(j1+m1)*fac(j1-m1)*fac(j2+m2)*fac(j2-m2)*fac(j+m)*fac(j-m)
        def D2(k):
            return fac(k)*fac(j1+j2-j-k)*fac(j1-m1-k)*fac(j2+m2-k)*fac(j-j2+m1+k)*fac(j-j1-m2+k)
        factor1 = self.delta(m1+m2-m)*scipy.sqrt(N1/float(D1))
        #find range for k, largest k such that no factorials are negative
        klist = scipy.rint(scipy.asarray([j1+j2-j,j1-m1,j2+m2]))
        kmax = int(klist.min())
        klist = scipy.rint(scipy.asarray([0,-(j-j2+m1),-(j-j1-m2)]))
        kmin = int(klist.max())
        factor2 = 0
        for k in range(kmin,kmax+1):
            factor2 += ((-1)**k)*scipy.sqrt(N2)/D2(k)
        return factor1*factor2

    def C(self,I,J):
        # makes a an array of cg coefs C s.t. fstates=C*ijstates
        fs,js = self.fstates,self.jstates
        C = []
        for (f,m) in fs:
            C.append([self.cg(J,mj,I,mi,f,m) for (mi,mj) in js])
        return scipy.asarray(C).transpose()

    def Hb_j(self,js):
        # zeeman interaction Hamiltonian in J basis in MHz
        H = [(self.gi*mi + self.gj*mj) for (mi,mj) in js]
        Hs = scipy.asarray(H)*mu_B
        return scipy.identity(self.N)*Hs*self.H_to_MHz

    def Hb_f(self):
        # zeeman interaction Hamiltonian in F basis in MHz
        a1 = scipy.dot(self.Hb_jbasis/self.H_to_MHz,self.cg_coef)
        Hf = scipy.dot(self.cg_coef.transpose(),a1)
        return Hf*self.H_to_MHz

    def Hhfs(self,fs):
        #calculate hyperfine engery shift diagonal in F basis
        L,S,I,J = self.LSIJ
        D = 2*I*(2*I-1)*2*J*(2*J-1)
        def K(f):
            return (f*(f+1)-I*(I+1)-J*(J+1))
        dEa = [ K(f) for (f,m) in fs]
        if D == 0:
            dEb = [0 for (f,m) in fs]
        else:   
            dEb = [(3*K(f)*(K(f)+1)/2-2*I*(I+1)*J*(J+1))/D for (f,m) in fs]
        dEA = scipy.asarray(dEa)*self.Ahfs/2.0
        dEB = scipy.asarray(dEb)*self.Bhfs
        return scipy.identity(self.N)*(dEA+dEB)*self.H_to_MHz

    def diagonalize(self,Bfield):
        # for a given B field in gauss diagonalize total H=Hhfs + Hb_f
        # returns energy eigen values in MHz sorted by
        Htot = (self.Hhfs_fbasis + Bfield*self.Hb_fbasis)
        [vals,vecs] = numpy.linalg.eig(Htot)
        self.evecs = vecs.transpose()
        self.evals = vals
        self.sorted_evals = self.sortid()
        return self.sorted_evals
    
    def getLabels_ij(self):
        '''
        gets labels for each state/curve in terms of mi,mj
        '''
        self.diagonalize(self.FINE_STRUCTURE_SEPARATION_FIELD)
        j = []
        self.sorted_ijstates = []
        for i in self.indices:#scipy.sort(self.indices):
            vj = scipy.inner(self.cg_coef,self.evecs[i])
            ji = scipy.argmax(vj*vj)
            mi,mj = self.jstates[ji]
            self.sorted_ijstates.append((mi,mj))
            s = '%d:: mi,mj = %s = %.1f'%(i,self.jstates[ji],mi+mj)
            j.append(s)
        return j

    def getLabels_fm(self):
        '''
        gets labels for each state/curve in terms of f,mf
        '''
        self.diagonalize(self.HYPERFINE_STRUCTURE_SEPARATION_FIELD)
        j = []
        self.sorted_fmstates = []
        for i in self.indices:#scipy.sort(self.indices):
            vj = self.evecs[i]
            fm = scipy.argmax(vj*vj)
            f,mf = self.fstates[fm]
            self.sorted_fmstates.append((f,mf))
            s = '%d:: f,mf = %s = %.1f'%(i,self.fstates[fm],mf)
            j.append(s)
        return j

    def get_E_from_fmf(self,(f,mf)):
        '''
        after diagonalizing at some B this will select
        value for a given (f,mf).
        '''
        i = self.sorted_fmstates.index((f,mf))
        return self.sorted_evals[i]

    def get_E_from_fmf_B(self,(f,mf),B):  #E in MHz=E/h
        self.diagonalize(B)
        E = self.get_E_from_fmf((f,mf))
        return E
        
    def getBcurve(self,Bvalues):
        self.energies = self.diagonalize(Bvalues[0])
        for b in Bvalues[1:]:
            vals = self.diagonalize(b)
            self.energies = numpy.column_stack([self.energies,vals])
        return self.energies

    def sortid(self):
        '''
        sort the eigenvalues into the 'adiabatic' states based on two criteria:
        a gross sort is performed by <m_f> for each eigenstate.  This is possible
        because mf = mi + mj is preserved and remains the same for each of these 
        states.  However there are several states with the same m_f from different
        f or j states.  We can take the eigenstates and find the expectation value of 
        mf for each of these and order the states by these mf values and then for the 
        states that have the same mf we will separate these as follows.
        We 'assume' that the states with the same m_f never cross.
        therefore m_f1 will always have a lesser evalue than m_f2 so that we can 
        break the degeneracy by adding a small value to the mf values proportional to 
        the evalues but much less than the typical m_f splitting (Order(1))
        So we use dx = evalues_mf/(evalue_range*100). (ie dx = [0 to .01])
        now we argsort(<mf>+dx) to get the arguments that would put these values
        in order (always the same order since dx will always be higher for one mf state) 
        and use them to order the eigen values then append this to the 
        values.  Note that this is the order of the jstates vector
        '''
        mf_values = [mf for (f,mf) in self.fstates]
        sortgroup1 = numpy.sum(self.evecs*self.evecs*mf_values,axis=1)
            #this gives <mf>q for each eigenvector Vq by <Vq|mf|Vq>
            # = sum(Cqi*Cqi*mfi) where sum over i is over the f states
            # where Vq = sum(Cqi*|f,mf>_i) hence Cqi is the eigenvector
            # matrix.transpose().  In numpy this sum is just C*C*mf_values
            # summed over the rows.  
        pos_vals = self.evals-self.evals.min()
        sortgroup2 = pos_vals/(100*pos_vals.max())
        sortvalues = sortgroup1 + sortgroup2
        self.indices = numpy.argsort(-1*sortvalues) # '-' to sort hi to lo
        vals = numpy.asarray([self.evals[i] for i in self.indices])
        return vals


class QuantumNumbers(object):
    '''
    this class is instantiated with a string identifying the isotope
    It provides a list of quantum numbers for that isotope.
    '''
    def __init__(self,s):
        try:
            self.m = getattr(self,s)
        except AttributeError:
            s2 = 'rb85_52s1_2'
            print '%s is not a preset isotope code: using %s instead'%(s,s2)
            self.m = getattr(self,s2)
    
    @property
    def numbers(self):
        return self.m

    @property
    def s1(self):
        return self.rb85_52s1_2

    @property
    def rb85_52s1_2(self):
        self.title = r'$\mathrm{Rb85}$'+'   '+r'$ 5^2S_{\frac{1}{2}}$'
        M = 85*u
        L = 0*1e-4 #1T
        S = 1/2.
        I = 5/2.
        J = 1/2.
        A = h*(1.011910813e9) # in J
        B = 0
        gi = -.000293 
        return (M,L,S,I,J,A,B,gi)

    @property
    def s2(self):
        return self.rb85_52p1_2

    @property
    def rb85_52p1_2(self):
        self.title = r'$\mathrm{Rb85}$'+'   '+r'$ 5^2P_{\frac{1}{2}}$'
        M = 85*u
        L = 1
        S = 1/2.
        I = 5/2.
        J = 1/2.
        A = h*(120.527e6) # in J
        B = 0
        gi = -.000293 
        return (M,L,S,I,J,A,B,gi)

    @property
    def s3(self):
        return self.rb85_52p3_2

    @property
    def rb85_52p3_2(self):
        self.title = r'$\mathrm{Rb85}$'+'   '+r'$ 5^2P_{\frac{3}{2}}$'
        M = 85*u
        L = 1
        S = 1/2.
        I = 5/2.
        J = 3/2.
        A = h*(25.002e6) # in J
        B = h*(25.790e6) # in J
        gi = -.000293 
        return (M,L,S,I,J,A,B,gi)
    @property
    def s4(self):
        return self.rb87_52s1_2

    @property
    def rb87_52s1_2(self):
        self.title = r'$\mathrm{Rb87}$'+'   '+r'$ 5^2S_{\frac{1}{2}}$'
        M = 87*u
        L = 0
        S = 1/2.
        I = 3/2.
        J = 1/2.
        A = h*(3.41734130545215e9) # in J
        B = 0
        gi = -.000995 
        return (M,L,S,I,J,A,B,gi)

    @property
    def s5(self):
        return self.rb87_52p1_2

    @property
    def rb87_52p1_2(self):
        self.title = r'$\mathrm{Rb87}$'+'   '+r'$ 5^2P_{\frac{1}{2}}$'
        M = 87*u
        L = 1
        S = 1/2.
        I = 3/2.
        J = 1/2.
        A = h*(408.328e6) # in J
        B = 0
        gi = -.000995 
        return (M,L,S,I,J,A,B,gi)

    @property
    def s6(self):
        return self.rb87_52p3_2

    @property
    def rb87_52p3_2(self):
        self.title = r'$\mathrm{Rb87}$'+'   '+r'$ 5^2P_{\frac{3}{2}}$'
        M = 87*u
        L = 1
        S = 1/2.
        I = 3/2.
        J = 3/2.
        A = h*(84.7185e6) # in J
        B = h*(12.4965e6) # in J
        gi = -.000995 
        return (M,L,S,I,J,A,B,gi)

    @property
    def s7(self):
        return self.li6_22s1_2

    @property
    def li6_22s1_2(self):
        self.title = r'$\mathrm{Li6}$'+'   '+r'$ 2^2S_{\frac{1}{2}}$'
        M = 6.015*u
        L = 0
        S = 1/2.
        I = 1.
        J = 1/2.
        A = h*(152.1368407e6) # in J
        B = 0 # in J
        gi = -.000447654
        return (M,L,S,I,J,A,B,gi)

    @property
    def s8(self):
        return self.li6_22p1_2

    @property
    def li6_22p1_2(self):
        self.title = r'$\mathrm{Li6}$'+'   '+r'$ 2^2P_{\frac{1}{2}}$'
        M = 6.015*u
        L = 0
        S = 1/2.
        I = 1.
        J = 1/2.
        A = h*(17.386e6) # in J
        B = 0 # in J
        gi = -.000447654
        return (M,L,S,I,J,A,B,gi)
        
    @property
    def s9(self):
        return self.li6_22p3_2

    @property
    def li6_22p3_2(self):
        self.title = r'$\mathrm{Li6}$'+'   '+r'$ 2^2P_{\frac{3}{2}}$'
        M = 6.015*u
        L = 0
        S = 1/2.
        I = 1.
        J = 3/2.
        A = h*(-1.155e6) # in J
        B = h*(-.1e6) # in J
        gi = -.000447654
        return (M,L,S,I,J,A,B,gi)

    @property
    def s10(self):
        return self.ca48_31d2

    @property
    def ca48_31d2(self):
        self.title = r'$\mathrm{Ca48}$'+'   '+r'$ 3^1D_{2}$'
        self.title = 'Ca48 3^1D_{2}'
        M = 47.95*u
        L = 2
        S = 0
        I = 0
        J = 2.
        A = h*(0) # in J
        B = h*(0) # in J
        gi = 0
        return (M,L,S,I,J,A,B,gi)

    @property
    def s11(self):
        return self.lu176_2d3_2

    @property
    def lu176_2d3_2(self):
        self.title = r'$Lu176$'+'   '+r'$ 3^1D_{2}$'
        self.title = 'Lu176 2D3_2'
        M = 176*u
        L = 2.
        S = .5
        I = 7.
        J = 1.5
        A = h*(0) # in J
        B = h*(0) # in J
        gi = 0
        return (M,L,S,I,J,A,B,gi)




if __name__ == "__main__":
    #select a state to view using Q(sx) defined as...
    s1='rb85_52s1_2'
    s2='rb85_52p1_2'
    s3='rb85_52p3_2'
    
    s4='rb87_52s1_2'
    s5='rb87_52p1_2'
    s6='rb87_52p3_2'
    
    s7='li6_22s1_2'
    s8='li6_22p1_2'
    s9='li6_22p3_2'
    s10='ca48_31d2'
    s11='lu176_2d3_2'
    
    Q = QuantumNumbers(s6)  #QuantumNumbers class instance for this isotope
    M,L,S,I,J,Ahfs,Bhfs,gi = Q.numbers
    Z = Zeeman(L,S,I,J,Ahfs,Bhfs,gi)  #Zeeman class instance for L,S,I,J,Ahfs,Bhfs
    print "g_j=",Z.gj
    print 'acc/(T/m)  =', mu_B*Z.gj/M
    #make plot
    pylab.figure(figsize=(12,8))
    #scipy.set_printoptions(precision=2)
    # get all state curves for B value range

    N=500
    Bmax = 200 # gauss
    j = range(1,N+1)
    jj = numpy.asarray(j)
    Bvals = Bmax*jj*(1/float(N))

    Z.getBcurve(Bvals*1e-4) #converted to Tesla
    for i in range(len(Z.energies)):
        Es = Z.energies[i]
        pylab.plot(Bvals,Es,label=Z.fmlabels[i]+Z.ijlabels[i]) 

    ## get a single (f,mf) state curve for given B value range
    #Es = []
    #state = (.5,.5)
    #for B in Bvals:
        #Es.append(Z.get_E_from_fmf_B(state,B))
    #numpy.asarray(Es)
    #pylab.plot(Bvals,Es,'rx',label=str(state))

    #label graph
    pylab.ylabel('Energy shift (MHz)')
    pylab.xlabel('B field (Gauss)')
    pylab.grid(True)
    pylab.xlim((-(Bmax/4.0),Bmax))
    leg=pylab.legend(loc=(0,0))#(1,.65)loc=3
    for t in leg.get_texts():
        t.set_fontsize(6)    # the legend text fontsize
    pylab.title(Q.title)
    pylab.show()
    
