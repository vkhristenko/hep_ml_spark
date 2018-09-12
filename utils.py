import numpy as np
import math
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

#import findspark
#findspark.init('/afs/cern.ch/work/m/migliori/public/spark2.3.1')

def PFIso(p, DR, PtMap, subtractPt):
    if p.Pt() <= 0.: return 0.
    DeltaEta = PtMap[:,0] - p.Eta()
    DeltaPhi = PtMap[:,1] - p.Phi()
    twopi = 2.*3.1415
    DeltaPhi = DeltaPhi - twopi*(DeltaPhi >  twopi) + twopi*(DeltaPhi < -1.*twopi)
    isInCone = DeltaPhi*DeltaPhi + DeltaEta*DeltaEta < DR*DR
    Iso = PtMap[isInCone, 2].sum()/p.Pt()
    if subtractPt: Iso = Iso -1
    return float(Iso)

# get the selected tracks
def ChPtMapp(DR, event):
    pTmap = []
    for h in event.EFlowTrack:
        if h.PT<= 0.5: continue
        pTmap.append([h.Eta, h.Phi, h.PT])
    return np.asarray(pTmap)

# get the selected neutrals
def NeuPtMapp(DR, event):
    pTmap = []
    #nParticles = 0
    for h in event.EFlowNeutralHadron:
        if h.ET<= 1.0: continue
        pTmap.append([h.Eta, h.Phi, h.ET])
    return np.asarray(pTmap)

# get the selected photons
def PhotonPtMapp(DR, event):
    pTmap = []
    for h in event.EFlowPhoton:
        if h.ET<= 1.0: continue
        pTmap.append([h.Eta, h.Phi, h.ET])
    return np.asarray(pTmap)

def Phi_mpi_pi(x):
    while x >= 3.1415: 
        x -= 2*3.1415
    while x < -3.1415:
        x += 2*3.1415
    return x

class LorentzVector(object):
    def __init__(self, *args):
        if len(args)>0:
            self.x = args[0]
            self.y = args[1]
            self.z = args[2]
            self.t = args[3]
    
    def SetPtEtaPhiM(self, pt, eta, phi, mass):
        pt = abs(pt)
        self.SetXYZM(pt*math.cos(phi), pt*math.sin(phi), pt*math.sinh(eta), mass)
        
    def SetXYZM(self, x, y, z, m):
        self.x = x;
        self.y = y
        self.z = z
        if (m>=0):
            self.t = math.sqrt(x*x + y*y + z*z + m*m)
        else:
            self.t = math.sqrt(max(x*x + y*y + z*z - m*m, 0))
            
    def E(self):
        return self.t
    
    def Px(self): 
        return self.x
    
    def Py(self):
        return self.y
    
    def Pz(self):
        return self.z
    
    def Pt(self):
        return math.sqrt(self.x*self.x + self.y*self.y)
    
    def Eta(self):
        cosTheta = self.CosTheta()
        if cosTheta*cosTheta<1:
            return -0.5*math.log((1.0 - cosTheta)/(1.0 + cosTheta))
        if self.z == 0: return 0
    
    def mag(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
    
    def CosTheta(self):
        return 1.0 if self.mag()==0.0 else self.z/self.mag()
    
    def Phi(self):
        return math.atan2(self.y, self.x)
    
    def DeltaR(self, other):
        deta = self.Eta() - other.Eta()
        dphi = Phi_mpi_pi(self.Phi() - other.Phi())
        return math.sqrt(deta*deta + dphi*dphi)
    
def mysign_func(v):
    if v<0: return -1.
    elif v==0: return 0.
    else: return 1.
    
#############################################
############################################

def selection(event, TrkPtMap, NeuPtMap, PhotonPtMap):
    # one electron or muon with pT> 15 GeV
    if event.Electron_size == 0 and event.MuonTight_size == 0: 
        return False, False, False
    foundMuon = None #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, 0, 0, 1, 1]
    foundEle =  None #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0, 0, 1, 0, 1]
    
    #
    # Has to replace the TLorentzVector functionality
    #
    l = LorentzVector()
    for ele in event.Electron:
        if ele.PT <= 25.: continue
        #
        # Has to replace the TLorentzVector functionality
        #
        l.SetPtEtaPhiM(ele.PT, ele.Eta, ele.Phi, 0.)
        
        pfisoCh = PFIso(l, 0.3, TrkPtMap, True)
        pfisoNeu = PFIso(l, 0.3, NeuPtMap, False)
        pfisoGamma = PFIso(l, 0.3, PhotonPtMap, False)
        if foundEle == None and (pfisoCh+pfisoNeu+pfisoGamma)<0.2:
            #foundEle.SetPtEtaPhiM(ele.PT, ele.Eta, ele.Phi, 0.)
            foundEle = [l.E(), l.Px(), l.Py(), l.Pz(), l.Pt(), l.Eta(), l.Phi(),
                        0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu,
                        0., 0., 0., 1., 0., float(ele.Charge)]
    for muon in event.MuonTight:
        if muon.PT <= 25.: continue
        #
        # Has to replace the TLorentzVector functionality
        #
        l.SetPtEtaPhiM(muon.PT, muon.Eta, muon.Phi, 0.)
        
        pfisoCh = PFIso(l, 0.3, TrkPtMap, True)
        pfisoNeu = PFIso(l, 0.3, NeuPtMap, False)
        pfisoGamma = PFIso(l, 0.3, PhotonPtMap, False)
        if foundMuon == None and (pfisoCh+pfisoNeu+pfisoGamma)<0.2:
            foundMuon = [l.E(), l.Px(), l.Py(), l.Pz(), l.Pt(), l.Eta(), l.Phi(),
                         0., 0., 0., pfisoCh, pfisoGamma, pfisoNeu,
                         0., 0., 0., 0., 1., float(muon.Charge)]
    if foundEle != None and foundMuon != None:
        if foundEle[5] > foundMuon[5]:
            return True, foundEle, foundMuon
        else:
            return True, foundMuon, foundEle
    if foundEle != None: return True, foundEle, foundMuon
    if foundMuon != None: return True, foundMuon, foundEle
    return False, None, None

#########################################
########################################

def convert(event):
    q = LorentzVector()
    particles = []
    TrkPtMap = ChPtMapp(0.3, event)
    NeuPtMap = NeuPtMapp(0.3, event)
    PhotonPtMap = PhotonPtMapp(0.3, event)
    if TrkPtMap.shape[0] == 0: return Row()
    if NeuPtMap.shape[0] == 0: return Row()
    if PhotonPtMap.shape[0] == 0: return Row()
    
    #
    # Get leptons
    #
    selected, lep, otherlep = selection(event, TrkPtMap, NeuPtMap, PhotonPtMap)
    if not selected: return Row()
    particles.append(lep)
    lepMomentum = LorentzVector(lep[1], lep[2], lep[3], lep[0])
    nTrk = 0
    
    #
    # Select Tracks
    #
    for h in event.EFlowTrack:
        if nTrk>=450: continue
        if h.PT<=0.5: continue
        q.SetPtEtaPhiM(h.PT, h.Eta, h.Phi, 0.)
        if lepMomentum.DeltaR(q) > 0.0001:
            pfisoCh = PFIso(q, 0.3, TrkPtMap, True)
            pfisoNeu = PFIso(q, 0.3, NeuPtMap, False)
            pfisoGamma = PFIso(q, 0.3, PhotonPtMap, False)
            particles.append([q.E(), q.Px(), q.Py(), q.Pz(),
                              h.PT, h.Eta, h.Phi, h.X, h.Y, h.Z,
                              pfisoCh, pfisoGamma, pfisoNeu,
                              1., 0., 0., 0., 0., mysign_func(h.PID)])
            nTrk += 1
    nPhoton = 0
    
    #
    # Select Photons
    #
    for h in event.EFlowPhoton:
        if nPhoton >= 150: continue
        if h.ET <= 1.: continue
        q.SetPtEtaPhiM(h.ET, h.Eta, h.Phi, 0.)
        pfisoCh = PFIso(q, 0.3, TrkPtMap, True)
        pfisoNeu = PFIso(q, 0.3, NeuPtMap, False)
        pfisoGamma = PFIso(q, 0.3, PhotonPtMap, False)
        particles.append([q.E(), q.Px(), q.Py(), q.Pz(),
                          h.ET, h.Eta, h.Phi, 0., 0., 0.,
                          pfisoCh, pfisoGamma, pfisoNeu,
                          0., 0., 1., 0., 0., 0.])
        nPhoton += 1
    nNeu = 0
    
    #
    # Select Neutrals
    #
    for h in event.EFlowNeutralHadron:
        if nNeu >= 200: continue
        if h.ET <= 1.: continue
        q.SetPtEtaPhiM(h.ET, h.Eta, h.Phi, 0.)
        pfisoCh = PFIso(q, 0.3, TrkPtMap, True)
        pfisoNeu = PFIso(q, 0.3, NeuPtMap, False)
        pfisoGamma = PFIso(q, 0.3, PhotonPtMap, False)
        particles.append([q.E(), q.Px(), q.Py(), q.Pz(),
                          h.ET, h.Eta, h.Phi, 0., 0., 0.,
                          pfisoCh, pfisoGamma, pfisoNeu,
                          0., 1., 0., 0., 0., 0.])
        nNeu += 1
    for iTrk in range(nTrk, 450):
        particles.append([0., 0., 0., 0., 0., 0., 0., 0.,0.,
                          0.,0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for iPhoton in range(nPhoton, 150):
        particles.append([0., 0., 0., 0., 0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for iNeu in range(nNeu, 200):
        particles.append([0., 0., 0., 0., 0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        
    #
    # High Level Features
    #
    myMET = event.MissingET[0]
    MET = myMET.MET
    phiMET = myMET.Phi
    MT = 2.*MET*lepMomentum.Pt()*(1-math.cos(lepMomentum.Phi()-phiMET))
    HT = 0.
    nJets = 0.
    nBjets = 0.
    for jet in event.Jet:
        if jet.PT > 40 and abs(jet.Eta)<2.4:
            nJets += 1
            HT += jet.PT
            if jet.BTag>0: 
                nBjets += 1
    LepPt = lep[4]
    LepEta = lep[5]
    LepPhi = lep[6]
    LepIsoCh = lep[10]
    LepIsoGamma = lep[11]
    LepIsoNeu = lep[12]
    LepCharge = lep[18]
    LepIsEle = lep[16]
    hlf = Vectors.dense([HT, MET, phiMET, MT, nJets, nBjets, LepPt, LepEta, LepPhi,
           LepIsoCh, LepIsoGamma, LepIsoNeu, LepCharge, LepIsEle])
    
        
    #
    # return the Row of low level features and high level features
    #
    
    return Row(lfeatures=particles, hfeatures=hlf, label=event.label)
