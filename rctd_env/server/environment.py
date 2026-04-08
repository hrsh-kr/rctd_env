"""
RCTD Environment — Core game logic and Epistemic Engine.

This is the heart of the environment. It implements:
  1. Deterministic scenario generation from seed
  2. Five distinct action types with budget costs
  3. Noise/reliability mechanics for evidence
  4. Rich reward shaping (terminal + step-wise)
  5. Comprehensive evaluation metrics on termination

Performance target: <1ms per step() call.
"""

from __future__ import annotations

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openenv.core import Environment

from ..models import (
    EvidenceItem,
    ExpertHint,
    RCTDAction,
    RCTDObservation,
    RCTDState,
)

# ═══════════════════════════════════════════════════════════════════════════
# Scenario Templates — 5 hard-coded, polished themes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioTemplate:
    """A real-world scenario theme for episode generation."""
    theme: str
    domain: str
    hypothesis_pool: List[str]
    evidence_templates: List[Dict[str, Any]]


SCENARIO_TEMPLATES: List[ScenarioTemplate] = [
    # -- 1. Medical Diagnosis ------------------------------------------------
    ScenarioTemplate(
        theme="medical_diagnosis",
        domain="Medicine",
        hypothesis_pool=[
            "The patient's condition is caused by a bacterial infection (Staphylococcus aureus bacteremia)",
            "The patient's condition is caused by a viral infection (Epstein-Barr virus mononucleosis)",
            "The patient's condition is caused by an environmental toxin (organophosphate poisoning)",
            "The patient's condition is caused by an autoimmune disorder (systemic lupus erythematosus)",
            "The patient's condition is caused by a genetic mutation (Factor V Leiden thrombophilia)",
        ],
        evidence_templates=[
            {"text": "CBC with differential: WBC 18,200/uL (ref: 4,500-11,000), neutrophils 82%, bands 12% indicating left shift. Platelet count 142,000/uL (mild thrombocytopenia). ESR 68 mm/hr.", "supports": [0, 1, 3], "contradicts": [], "base_confidence": 0.7},
            {"text": "Blood culture (2 of 2 sets): Gram-positive cocci in clusters growing at 14 hours. Preliminary identification pending MALDI-TOF confirmation. Sensitivity panel ordered.", "supports": [0], "contradicts": [1, 2, 4], "base_confidence": 0.88},
            {"text": "Procalcitonin level: 8.4 ng/mL (ref: <0.1 ng/mL). CRP 186 mg/L. Lactate 3.2 mmol/L. qSOFA score: 2 (altered mentation + RR 24).", "supports": [0], "contradicts": [4], "base_confidence": 0.82},
            {"text": "Monospot test (heterophile antibody): Positive. Peripheral smear shows 38% atypical lymphocytes. EBV VCA IgM titer elevated at 1:320.", "supports": [1], "contradicts": [0, 2], "base_confidence": 0.75},
            {"text": "Red blood cell cholinesterase level: 2,100 U/L (ref: 5,300-12,900 U/L), consistent with 60% enzyme inhibition. Serum cholinesterase similarly depressed at 1,800 U/L.", "supports": [2], "contradicts": [0, 1, 3, 4], "base_confidence": 0.85},
            {"text": "Patient presents with SLUDGE syndrome: salivation, lacrimation, urination, defecation, GI distress, emesis. Pupils 2mm bilaterally (miosis). Fasciculations noted in deltoids.", "supports": [2], "contradicts": [1, 4], "base_confidence": 0.78},
            {"text": "ANA titer: 1:640, homogeneous pattern. Anti-dsDNA antibodies: 240 IU/mL (ref: <30). Complement C3: 52 mg/dL (low), C4: 8 mg/dL (low). Urinalysis: proteinuria 2+, RBC casts.", "supports": [3], "contradicts": [0, 2], "base_confidence": 0.80},
            {"text": "CT chest shows bilateral ground-glass opacities with subpleural sparing. No cavitation or lobar consolidation. Mediastinal lymphadenopathy present (largest node 2.1cm).", "supports": [1, 3], "contradicts": [], "base_confidence": 0.55},
            {"text": "Genetic panel (thrombophilia workup): Factor V Leiden mutation detected (heterozygous, R506Q). Protein C activity normal at 118%. MTHFR C677T variant: heterozygous.", "supports": [4], "contradicts": [0, 1, 2], "base_confidence": 0.90},
            {"text": "Blood smear peripheral morphology: Toxic granulation and Dohle bodies in neutrophils. No schistocytes or spherocytes. Reticulocyte count 1.8% (normal). Haptoglobin 145 mg/dL (normal), ruling out hemolytic process.", "supports": [0], "contradicts": [3, 4], "base_confidence": 0.72},
            {"text": "Lumbar puncture results: CSF WBC 2 cells/uL (normal), protein 38 mg/dL (normal), glucose 62 mg/dL (normal). No organisms on Gram stain. CSF HSV PCR negative. Opening pressure 14 cmH2O.", "supports": [0, 3], "contradicts": [1], "base_confidence": 0.60},
            {"text": "Urine toxicology screen (GC-MS): Positive for organophosphate metabolites (diethylphosphate 142 ug/L, diethylthiophosphate 89 ug/L). Serum atropine challenge test: marked improvement in secretions within 5 minutes of 2mg IV atropine.", "supports": [2], "contradicts": [0, 1, 3, 4], "base_confidence": 0.87},
            {"text": "Echocardiogram (TTE): No vegetations on valves (Duke criteria not met). LVEF 58% (normal). No pericardial effusion. Aortic and mitral valves morphologically normal. Rules out infective endocarditis as source of bacteremia.", "supports": [1, 3], "contradicts": [0], "base_confidence": 0.58},
            {"text": "Skin biopsy (punch, 4mm, malar area): Interface dermatitis with vacuolar changes at DEJ. Mucin deposition in dermis. Direct immunofluorescence: granular IgG and C3 deposits at basement membrane zone (lupus band positive).", "supports": [3], "contradicts": [0, 1, 2, 4], "base_confidence": 0.86},
            {"text": "Patient history: 34F, no chemical exposure, no travel, no family history of clotting disorders. Symptoms: 5-day fever (39.2C), fatigue, malar rash, arthralgia in MCP joints bilaterally.", "supports": [1, 3], "contradicts": [2, 4], "base_confidence": 0.65},
        ],
    ),
    # -- 2. Market Disruption Analysis ---------------------------------------
    ScenarioTemplate(
        theme="market_analysis",
        domain="Finance",
        hypothesis_pool=[
            "The market disruption is driven by a supply-chain shock (Suez Canal blockage + Red Sea rerouting)",
            "The market disruption is driven by a demand-side shift (post-pandemic remote work reducing commercial real estate)",
            "The market disruption is driven by new regulatory policy (EU AI Act compliance costs)",
            "The market disruption is driven by a competitor's technological breakthrough (quantum computing milestone)",
            "The market disruption is driven by macroeconomic currency fluctuations (USD/CNY intervention)",
        ],
        evidence_templates=[
            {"text": "Baltic Dry Index surged 47% in 3 weeks (from 1,240 to 1,824). Drewry World Container Index shows 40-foot container rates Shanghai-Rotterdam at $4,890 (up from $1,420 baseline). Maersk issued force majeure on 12 routes.", "supports": [0], "contradicts": [1], "base_confidence": 0.82},
            {"text": "Kastle Systems back-to-work barometer: US office occupancy at 47.3% (pre-pandemic: 95%). WeWork filed Chapter 11. Cushman and Wakefield reports 950M sq ft of vacant US office space (18.2% vacancy, highest since 1991).", "supports": [1], "contradicts": [], "base_confidence": 0.75},
            {"text": "European Commission published final EU AI Act implementing rules. Compliance deadline: Aug 2025 for prohibited systems. Estimated industry compliance cost: 31B EUR across EU tech sector. GDPR-style extraterritorial scope confirmed.", "supports": [2], "contradicts": [], "base_confidence": 0.70},
            {"text": "IBM announced 1,121-qubit Condor processor achieving quantum volume 2^18. Google DeepMind published fault-tolerant error correction below threshold. D-Wave stock +340% in 60 days. McKinsey estimates $1.3T quantum market by 2035.", "supports": [3], "contradicts": [], "base_confidence": 0.68},
            {"text": "PBOC set USD/CNY midpoint at 7.1088, strongest fix in 4 months. FX reserves drew down $26.8B. Offshore CNH implied vol spiked to 8.2%. Carry trade unwind in AUD/JPY triggered margin calls across Asian desks.", "supports": [4], "contradicts": [1, 3], "base_confidence": 0.73},
            {"text": "Port of Rotterdam throughput data: TEU volume down 28% YoY. Average vessel waiting time increased from 1.2 to 6.8 days. Insurance premiums for Red Sea transit up 400%. Alternative Cape of Good Hope routing adds 10-14 days.", "supports": [0], "contradicts": [1, 2, 3, 4], "base_confidence": 0.80},
            {"text": "Bloomberg terminal data: Sector rotation analysis shows Tech (-8.2%), Real Estate (-12.4%), Industrials (+3.1%), Utilities (+5.7%). VIX at 28.4 (elevated). Put/call ratio on SPY at 1.34.", "supports": [1, 2, 3], "contradicts": [], "base_confidence": 0.50},
            {"text": "Federal Reserve Beige Book: 8 of 12 districts report slight to modest growth deceleration. Labor market cooling: JOLTS openings down to 8.1M from 10.7M peak. Core PCE sticky at 3.2%.", "supports": [4], "contradicts": [3], "base_confidence": 0.60},
            {"text": "Reuters survey of 40 industry analysts: 62% cite regulatory uncertainty as primary headwind. STOXX Europe 600 Tech sub-index underperforming broader index by 340bps since Act announcement.", "supports": [2], "contradicts": [0], "base_confidence": 0.65},
            {"text": "Shipping satellite AIS data (MarineTraffic): 47 container vessels currently holding at Bab el-Mandab strait anchorage. Average wait time 8.4 days (normal: <1 day). CMA CGM and Hapag-Lloyd have rerouted 100% of Asia-Europe services via Cape of Good Hope since Jan 15.", "supports": [0], "contradicts": [1, 3, 4], "base_confidence": 0.83},
            {"text": "Commercial real estate REIT earnings (Q3): Boston Properties FFO down 14% YoY. Vornado Realty announced conversion of 3 office towers to residential. National office lease renewal rate: 61% (pre-pandemic: 82%). Sublease availability at 10-year high.", "supports": [1], "contradicts": [0], "base_confidence": 0.77},
            {"text": "SEC 8-K filing analysis: 23 EU-headquartered tech companies disclosed material AI Act compliance costs in recent filings. Average estimated spend: $45M for large-cap, $8M for mid-cap. Three companies announced relocation of AI research divisions to non-EU jurisdictions.", "supports": [2], "contradicts": [0, 4], "base_confidence": 0.72},
            {"text": "Semiconductor supply chain data (SEMI): Global wafer shipments up 12% QoQ. TSMC utilization rate at 94%. No reported shortages in consumer electronics or automotive chips. Lead times for standard components returned to pre-pandemic levels of 12-14 weeks.", "supports": [1, 2, 3, 4], "contradicts": [0], "base_confidence": 0.68},
            {"text": "Consumer confidence index (Conference Board): Present Situation Index fell to 134.5 (from 147.2). Expectations Index at 72.8 (recession signal <80). Labor differential (jobs plentiful minus hard to get) narrowed to 12.7 points, lowest since 2021.", "supports": [1, 4], "contradicts": [3], "base_confidence": 0.63},
            {"text": "Patent filing data (WIPO): Quantum computing filings up 89% YoY. Top filer shifted from IBM to undisclosed Chinese entity. VC funding in quantum startups: $2.3B in Q3 (vs $800M same quarter prior year).", "supports": [3], "contradicts": [0, 2, 4], "base_confidence": 0.62},
        ],
    ),
    # -- 3. Cybersecurity Incident -------------------------------------------
    ScenarioTemplate(
        theme="security_incident",
        domain="Cybersecurity",
        hypothesis_pool=[
            "The breach was caused by an insider threat (malicious employee with privileged access)",
            "The breach was caused by an external APT group (MITRE: APT29/Cozy Bear, Russian SVR)",
            "The breach was caused by a zero-day exploit (CVE-2024-3094: XZ Utils backdoor variant)",
            "The breach was caused by a social engineering / spearphishing campaign",
            "The breach was caused by misconfigured cloud infrastructure (AWS S3 + IAM policy)",
        ],
        evidence_templates=[
            {"text": "SIEM alert: Anomalous data exfiltration -- 2.4TB transferred to external IP 185.220.101.x (Tor exit node) between 02:00-04:00 UTC via DNS tunneling (TXT record queries at 847/min to randomized subdomains of legit-update[.]com).", "supports": [0, 1], "contradicts": [4], "base_confidence": 0.72},
            {"text": "CrowdStrike Falcon EDR: Process injection detected -- explorer.exe spawned rundll32.exe loading unsigned DLL from %APPDATA%. Behavioral signature matches MITRE ATT&CK T1055.001 (Process Hollowing). Cobalt Strike beacon config extracted: watermark 0x5109bf4d.", "supports": [1], "contradicts": [0, 4], "base_confidence": 0.80},
            {"text": "Vulnerability scan results: 14 internet-facing hosts running XZ Utils 5.6.0/5.6.1 (affected versions). OpenSSH with systemd integration confirmed on 9 hosts. Liblzma backdoor allows pre-auth RCE via crafted SSH certificates. CVSS 10.0.", "supports": [2], "contradicts": [0, 3], "base_confidence": 0.85},
            {"text": "Email gateway logs: 23 employees in Finance/Engineering received emails from spoofed DocuSign sender (envelope-from: noreply@docusign-verify[.]net). 7 clicked through. Credential harvesting page hosted on Cloudflare Workers. OAuth tokens for 4 Microsoft 365 accounts compromised.", "supports": [3], "contradicts": [], "base_confidence": 0.70},
            {"text": "AWS CloudTrail: S3 bucket prod-customer-data-2024 had BlockPublicAccess disabled on 2024-01-15 by IAM role deploy-automation. ListBucket and GetObject calls from 94 unique IPs across 12 countries in subsequent 72 hours. No bucket policy restricting access.", "supports": [4], "contradicts": [1, 2], "base_confidence": 0.88},
            {"text": "HR records cross-reference: Employee J.M. (Senior DevOps, termination date: 2024-02-01) retained VPN access for 18 days post-termination. Badge access logs show after-hours entry to server room on 3 dates. GitHub audit log: 4 repos cloned to personal device on final day.", "supports": [0], "contradicts": [1, 2, 4], "base_confidence": 0.75},
            {"text": "Threat intel correlation (VirusTotal + MISP): Malware sample SHA256:a3b5... matches YARA rule APT29_WellMess_Loader. Infrastructure overlap with 2020 SolarWinds campaign (shared SSL certificate CN). TTPs consistent with SVR targeting pattern.", "supports": [1], "contradicts": [0, 3, 4], "base_confidence": 0.78},
            {"text": "Forensic disk image analysis: Timeline shows initial compromise at 2024-01-22T03:41Z. Auth.log: successful SSH login using ed25519 key not in authorized_keys (backdoor injection). Process tree: sshd spawned unsigned .libsystemd.so (92KB) then reverse shell.", "supports": [2], "contradicts": [3, 4], "base_confidence": 0.83},
            {"text": "DLP alert review: 340 files matching PII regex (SSN, credit card) were accessed by service account svc-reporting which normally queries only aggregated tables. Access originated from J.M. last-known workstation IP. No MFA prompt triggered (service accounts exempt).", "supports": [0], "contradicts": [1, 2], "base_confidence": 0.68},
            {"text": "Active Directory audit log: Service account svc-backup had its password reset by J.M.'s admin credentials 72 hours before termination. Same service account was used for lateral movement to 3 database servers containing customer PII. No helpdesk ticket corresponds to this password reset.", "supports": [0], "contradicts": [1, 2, 4], "base_confidence": 0.82},
            {"text": "Mandiant threat intelligence report: APT29 shifted to cloud-focused operations in 2024. New tactic: abuse of trusted cloud services (Azure, AWS) for C2. No direct evidence of on-premise intrusion in this campaign. Targeting profile: government contractors and think tanks.", "supports": [1], "contradicts": [0, 4], "base_confidence": 0.70},
            {"text": "Patch management audit (Qualys): 67% of external-facing systems had critical patches applied within SLA. XZ Utils specifically was patched on 14 of 14 affected hosts within 48 hours of CVE publication. However, backdoor may have been active for up to 3 weeks before CVE disclosure.", "supports": [2], "contradicts": [], "base_confidence": 0.65},
            {"text": "Microsoft 365 audit log: Compromised OAuth tokens were used to create 3 new mail flow rules forwarding all C-suite email to external address. Rules created at 14:22 UTC using legitimate user session from residential ISP IP. Geo-impossible travel detected: London (14:00) -> Lagos (14:22).", "supports": [3], "contradicts": [0, 2], "base_confidence": 0.76},
            {"text": "AWS Config timeline: IAM policy arn:aws:iam::123456789:policy/S3FullAccess was attached to 8 additional roles between Jan 10-15. Change was made via Terraform by the deploy-automation role. No code review or approval in GitHub for corresponding IaC changes.", "supports": [4], "contradicts": [1, 2], "base_confidence": 0.79},
            {"text": "Network forensics: TLS certificate analysis on C2 channel shows Lets Encrypt cert issued 48hrs before breach. JA3 fingerprint matches known Cobalt Strike 4.9 profile. Beacon interval: 60s with 15% jitter. Metadata exfil via DNS over HTTPS to Cloudflare resolver.", "supports": [1, 2, 3], "contradicts": [4], "base_confidence": 0.65},
        ],
    ),
    # -- 4. Climate Event Attribution ----------------------------------------
    ScenarioTemplate(
        theme="climate_attribution",
        domain="Environmental Science",
        hypothesis_pool=[
            "The extreme weather event is primarily driven by natural oceanic cycles (strong El Nino, ONI +2.1C)",
            "The extreme weather event is primarily driven by anthropogenic greenhouse gas emissions (CO2 at 425.4 ppm)",
            "The extreme weather event is primarily driven by volcanic aerosol forcing (Hunga Tonga water vapor injection)",
            "The extreme weather event is primarily driven by urban heat island effects (megacity microclimate)",
            "The extreme weather event is primarily driven by regional deforestation (Amazon/Southeast Asia land-use change)",
        ],
        evidence_templates=[
            {"text": "NOAA Oceanic Nino Index (ONI): +2.1C for DJF 2023-24, qualifying as very strong El Nino. Subsurface Kelvin wave analysis shows warm pool extending to 150W. Trade wind anomaly: -4.2 m/s. Historical analog years (1997-98, 2015-16) show similar teleconnection patterns.", "supports": [0], "contradicts": [3], "base_confidence": 0.78},
            {"text": "Mauna Loa Observatory CO2: 425.4 ppm (May 2024), annual increase +3.4 ppm (highest recorded rate). Methane at 1,923 ppb (+14 ppb YoY). Global mean surface temperature anomaly: +1.48C above 1850-1900 baseline (ERA5 reanalysis).", "supports": [1], "contradicts": [], "base_confidence": 0.85},
            {"text": "NASA SAGE III/ISS stratospheric aerosol data: Aerosol optical depth (AOD) at 0.010 above baseline at 525nm in Southern Hemisphere. Hunga Tonga-Hunga Haapai injected estimated 146 Tg of water vapor into the stratosphere (January 2022). Residence time models suggest perturbation persists through 2025.", "supports": [2], "contradicts": [0], "base_confidence": 0.72},
            {"text": "Urban weather station network (EPA AirNow + local ASOS): City center station reads 4.8C above rural reference station 35km NW. Landsat 9 thermal band shows surface temperature differential of 6.2C between CBD impervious surfaces and surrounding agricultural land. Effect is nocturnal-dominant.", "supports": [3], "contradicts": [0, 1, 2], "base_confidence": 0.65},
            {"text": "Global Forest Watch / PRODES data: Amazon deforestation rate 11,568 km2 in 2023 (up 22% from 2019 baseline). Southeast Asia palm oil expansion: 2.4M hectares converted 2020-2024. Albedo change from forest to pasture estimated at +0.03 (shortwave), reducing evapotranspiration by 40%.", "supports": [4], "contradicts": [], "base_confidence": 0.68},
            {"text": "ECMWF ERA5 reanalysis: SST anomaly pattern shows canonical El Nino spatial signature in Nino 3.4 region. 200hPa velocity potential anomalies consistent with enhanced Walker circulation weakening. MJO phase analysis shows suppressed convection over Maritime Continent.", "supports": [0], "contradicts": [1, 3, 4], "base_confidence": 0.75},
            {"text": "IPCC AR6 attribution framework applied: Probability of this event magnitude under pre-industrial conditions <0.3%. Under current forcing: approximately 4% annually. Fraction of Attributable Risk (FAR) = 0.93 for anthropogenic warming. Multi-model ensemble (CMIP6) agreement: 38 of 42 models reproduce the event under SSP2-4.5.", "supports": [1], "contradicts": [0], "base_confidence": 0.82},
            {"text": "Radiosonde network (IGRA): Temperature profile shows stratospheric cooling (-0.8C at 50hPa) concurrent with tropospheric warming (+1.2C at 500hPa). This vertical signature is fingerprint of greenhouse gas forcing, not solar or volcanic (which warm the stratosphere).", "supports": [1], "contradicts": [2, 3], "base_confidence": 0.80},
            {"text": "CALIPSO lidar backscatter profiles: Enhanced aerosol layer detected at 28-32km altitude, consistent with Hunga Tonga volcanic plume trajectory modeling. Particle size distribution peaks at 0.5um. No comparable stratospheric perturbation since Pinatubo (1991).", "supports": [2], "contradicts": [0, 4], "base_confidence": 0.77},
            {"text": "Ocean buoy network (TAO/TRITON array): Subsurface temperature at 150m depth along equatorial Pacific shows +4.2C anomaly. Warm water volume (>20C isotherm) expanded to 135% of climatological mean. Thermocline depth anomaly: +28m. Pattern is textbook El Nino Modoki (central Pacific type).", "supports": [0], "contradicts": [3, 4], "base_confidence": 0.80},
            {"text": "Ice core proxy data (Vostok, EPICA): Current CO2 level (425 ppm) exceeds any value in the 800,000-year record. Rate of increase (3.4 ppm/yr) is 10x faster than the fastest natural increase during deglaciation events. Methane similarly unprecedented at 1,923 ppb.", "supports": [1], "contradicts": [], "base_confidence": 0.88},
            {"text": "SO2 monitoring (Aura/OMI satellite): No significant SO2 plumes detected in the past 6 months. Stratospheric SO2 column density at background levels (0.1 DU). No volcanic eruptions with VEI >= 4 since Hunga Tonga. Rules out fresh volcanic aerosol injection.", "supports": [0, 1, 4], "contradicts": [2], "base_confidence": 0.73},
            {"text": "Microclimate sensor network (urban/rural paired stations): Temperature differential exhibits strong diurnal cycle -- UHI effect peaks at 01:00-04:00 local (delta: +5.8C) and minimizes at 14:00 (delta: +1.2C). Weekend vs weekday difference: 0.4C (anthropogenic heat flux signature).", "supports": [3], "contradicts": [], "base_confidence": 0.62},
            {"text": "GRACE-FO satellite gravimetry: Terrestrial water storage in Amazon basin decreased by 34 km3/year over 2020-2024 period. Evapotranspiration reduction detected via MODIS ET product. Regional precipitation recycling ratio dropped from 0.35 to 0.28 (moisture recycling breakdown).", "supports": [4], "contradicts": [3], "base_confidence": 0.74},
            {"text": "Flux tower network (FLUXNET): Latent heat flux at deforested Amazon sites reduced by 42% compared to paired intact forest references. Sensible heat flux increased proportionally. Regional moisture recycling model shows 18% reduction in recycled precipitation for downwind regions.", "supports": [4], "contradicts": [3], "base_confidence": 0.70},
        ],
    ),
    # -- 5. Historical Artifact Authentication -------------------------------
    ScenarioTemplate(
        theme="artifact_authentication",
        domain="Archaeology",
        hypothesis_pool=[
            "The artifact is an authentic piece from the claimed 4th century BCE Hellenistic period",
            "The artifact is a modern forgery (post-1950) created with period-appropriate materials",
            "The artifact is genuine antiquity but misdated -- actually from the 2nd century CE Roman period",
            "The artifact is a 19th-century museum replica made for educational display",
            "The artifact is a composite: authentic ancient fragments with modern restoration/assembly",
        ],
        evidence_templates=[
            {"text": "Radiocarbon dating (AMS, Beta Analytic Lab): Organic residue in surface patina yields 2,340 +/- 40 BP (calibrated 2-sigma range: 520-380 BCE). delta-13C = -24.3 per mil, consistent with C3 plant-derived binding medium. Sample weight: 12.4mg.", "supports": [0], "contradicts": [1, 3], "base_confidence": 0.82},
            {"text": "Scanning electron microscopy (SEM-EDS): Tool marks on base show parallel striations at 0.8mm spacing with irregular depth -- consistent with hand-worked bronze chisel. No evidence of rotary tool marks (which would indicate post-1800 manufacture). Surface crystallization: cuprite penetration to 180um depth.", "supports": [0, 2], "contradicts": [1, 3], "base_confidence": 0.76},
            {"text": "X-ray fluorescence (XRF) compositional analysis: Cu 88.2%, Sn 9.8%, Pb 1.4%, As 0.3%, Fe 0.2%, trace Ag/Sb. The arsenic-tin bronze composition is consistent with Eastern Mediterranean production c. 400-100 BCE. Lead isotope ratios (206/204: 18.72, 207/204: 15.66) match Laurion mines (Attica).", "supports": [0, 2], "contradicts": [1, 3], "base_confidence": 0.78},
            {"text": "Thermoluminescence (TL) dating of clay core fragments: Last firing event dated to 1,890 +/- 120 years before present. This places manufacture in the 1st-2nd century CE range -- approximately 400 years later than the claimed Hellenistic date.", "supports": [2], "contradicts": [0, 1, 3], "base_confidence": 0.84},
            {"text": "Provenance research: Object appears in the 1897 Baron Rothschild collection catalog (item #447, described as Hellenistic bronze, provenance unknown). No documentation between supposed antiquity and 1897. Geneva Freeport storage records: 1962-2019. No export license from country of origin.", "supports": [0, 2, 3], "contradicts": [], "base_confidence": 0.55},
            {"text": "UV fluorescence examination: Modern synthetic adhesive (cyanoacrylate signature) detected at 3 join points. Surrounding patina is disrupted at joins -- sharp boundary between weathered and unweathered surfaces. Two fragments show different patina coloration (olive-green vs. blue-green).", "supports": [4], "contradicts": [0, 1, 3], "base_confidence": 0.80},
            {"text": "Micro-CT scan (160kV, 12um resolution): Internal structure shows 2 distinct casting events -- left arm and torso have different internal porosity patterns and wall thicknesses (2.1mm vs 3.4mm). Core material: 2 different clay types visible in density mapping.", "supports": [4], "contradicts": [0, 1, 3], "base_confidence": 0.82},
            {"text": "Iconographic analysis (Dr. M. Korres, Athens): Hairstyle matches melon coiffure type common 340-300 BCE. Drapery folds use motion lines technique from Lysippan school. However, proportional system (7.5 head-heights) is characteristic of Roman copies of Greek originals.", "supports": [0, 2], "contradicts": [1, 3], "base_confidence": 0.60},
            {"text": "Corrosion product analysis (XRD): Primary patina layer is malachite over cuprite -- consistent with centuries of burial in calcareous soil. However, atacamite detected only on left arm fragment, suggesting different burial environment or post-excavation treatment.", "supports": [4], "contradicts": [1], "base_confidence": 0.72},
            {"text": "Metallographic cross-section (optical microscopy, 200x): Alpha-phase bronze with equiaxed grain structure (avg grain size 45um) and annealing twins -- consistent with repeated hot-working and annealing cycles used in ancient lost-wax casting. No cold-rolled microstructure.", "supports": [0, 2], "contradicts": [1, 3], "base_confidence": 0.74},
            {"text": "Accelerator mass spectrometry (AMS) of core clay organic inclusions: Radiocarbon age 1,950 +/- 55 BP (calibrated: 20-130 CE). This is statistically inconsistent with the claimed 4th century BCE date at >99% confidence. Consistent with Roman Imperial period production.", "supports": [2], "contradicts": [0, 1, 3], "base_confidence": 0.86},
            {"text": "Portable X-ray diffraction (p-XRD) of blue pigment on surface: Egyptian Blue (CaCuSi2O6) identified. This pigment was used extensively from 3rd millennium BCE through 4th century CE. Its presence is consistent with both Hellenistic and Roman dates but rules out 19th century manufacture.", "supports": [0, 2], "contradicts": [1, 3], "base_confidence": 0.68},
            {"text": "Inscription analysis (Dr. A. Matthaiou, epigraphist): Letter forms show sigma with 4 bars (archaic) but omega with modern proportions (post-3rd century BCE). Mixed chronological indicators suggest either transitional period (late 4th century) or deliberate archaizing in Roman era.", "supports": [0, 2], "contradicts": [1, 3], "base_confidence": 0.58},
            {"text": "3D surface scanning (structured light, 50um accuracy): Left arm attachment surface shows fresh fracture faces without patination, surrounded by artificial patina (copper chloride solution application detected by FTIR). Attachment method uses modern stainless steel dowel pins (316L grade).", "supports": [4], "contradicts": [0, 2], "base_confidence": 0.84},
            {"text": "Neutron activation analysis (NAA): Rare earth element (REE) profile of core clay -- La/Sm ratio 4.2, Eu anomaly 0.71 -- matches Corinthian clay sources documented in MURR database. This clay source was used in both Hellenistic and Roman periods. No modern clay signatures (e.g., kaolin processing markers) detected.", "supports": [0, 2], "contradicts": [1, 3], "base_confidence": 0.70},
        ],
    ),
    # -- 6. Legal Contract Dispute -----------------------------------------
    ScenarioTemplate(
        theme="legal_dispute",
        domain="Law",
        hypothesis_pool=[
            "The contract breach was caused by intentional fraud (deliberate misrepresentation of deliverables)",
            "The contract breach was caused by gross negligence (failure to exercise reasonable care in execution)",
            "The contract breach resulted from force majeure (unforeseeable external event beyond party control)",
            "The contract breach was caused by a clerical/drafting error (ambiguous contract language)",
            "The contract breach was caused by third-party interference (tortious interference with contract)",
        ],
        evidence_templates=[
            {"text": "Email discovery (Exhibit A-14): Internal memo from respondent's VP of Sales dated 3 months before contract signing: 'We cannot deliver the Q4 volume at this price point. Proceed with signing anyway -- we will renegotiate or claim supply issues later.'", "supports": [0], "contradicts": [1, 2, 3], "base_confidence": 0.88},
            {"text": "Forensic accounting report (Deloitte): Respondent's COGS for the contracted product were 340% of the contract price. Margin analysis shows the contract was economically unviable at inception. No evidence of post-signing cost increases.", "supports": [0], "contradicts": [2], "base_confidence": 0.82},
            {"text": "Expert witness testimony (Prof. J. Richardson, Contract Law): The force majeure clause in Section 14.2 requires 'unforeseeable circumstances beyond reasonable control.' The cited supply disruption was widely reported 6 weeks before contract execution.", "supports": [0, 1], "contradicts": [2], "base_confidence": 0.75},
            {"text": "Insurance claim records: Respondent filed a business interruption claim citing the same supply disruption. Insurer denied claim, stating 'The disruption was foreseeable and the insured failed to take reasonable mitigation steps.' Denial letter dated 2 weeks before breach notification.", "supports": [1], "contradicts": [2], "base_confidence": 0.70},
            {"text": "Contract markup analysis (redline comparison): Section 7.3 contains a decimal point error -- '$14.50 per unit' in the signed version vs '$145.00 per unit' in all prior drafts. Neither party's legal counsel flagged the discrepancy during review.", "supports": [3], "contradicts": [0], "base_confidence": 0.78},
            {"text": "Deposition transcript (respondent's CEO): 'We fully intended to perform. The supply chain disruption was unprecedented and made performance commercially impracticable.' Cross-examination revealed CEO was aware of disruption risk per industry reports.", "supports": [1, 2], "contradicts": [0], "base_confidence": 0.55},
            {"text": "Third-party communications (subpoenaed): Competitor XYZ Corp sent formal offer to respondent's key supplier, offering 200% premium for exclusive supply rights. Supplier confirmed diversion of materials. XYZ Corp had prior knowledge of the contract.", "supports": [4], "contradicts": [0, 1, 2], "base_confidence": 0.80},
            {"text": "Industry benchmarking data (IBISWorld): 73% of comparable contracts executed in the same period were fulfilled on time. Supply chain disruption affected only 12% of the specific material category. Market conditions were within normal variance.", "supports": [0, 1], "contradicts": [2], "base_confidence": 0.72},
            {"text": "Contract execution records: Both parties used the same law firm (conflict-of-interest waiver signed). Junior associate drafted final version. Senior partner review was skipped due to year-end rush. No independent legal review by either party.", "supports": [3], "contradicts": [0, 4], "base_confidence": 0.65},
            {"text": "Bank records (Exhibit F-7): Respondent transferred $2.4M to an offshore subsidiary 48 hours before sending breach notification. Transfer memo: 'Asset protection -- anticipated litigation.' Funds were earmarked for the contract's performance budget.", "supports": [0], "contradicts": [1, 2, 3], "base_confidence": 0.85},
        ],
    ),
    # -- 7. Epidemiological Outbreak Investigation --------------------------
    ScenarioTemplate(
        theme="outbreak_investigation",
        domain="Epidemiology",
        hypothesis_pool=[
            "The outbreak is caused by a foodborne pathogen (Salmonella enteritidis from contaminated poultry supply chain)",
            "The outbreak is caused by waterborne contamination (Cryptosporidium from municipal treatment failure)",
            "The outbreak is caused by an airborne respiratory pathogen (novel influenza A H5N1 reassortant)",
            "The outbreak is caused by a vector-borne disease (Aedes aegypti-transmitted Dengue serotype 3)",
            "The outbreak is caused by person-to-person contact transmission (Norovirus GII.4 Sydney variant)",
        ],
        evidence_templates=[
            {"text": "Case-control study (n=340, CDC EIS team): Cases 8.2x more likely to have eaten at restaurant chain X in 72hr window (OR=8.2, 95% CI: 4.1-16.3, p<0.001). Dose-response relationship with chicken salad consumption. Attack rate at implicated locations: 34%.", "supports": [0], "contradicts": [1, 3], "base_confidence": 0.85},
            {"text": "Municipal water testing (EPA Method 1623.1): Cryptosporidium oocysts detected at 4.2 oocysts/L in finished water (MCL: 0 per treatment technique). Turbidity exceedance logged at Plant B on days 3-5 of outbreak. UV disinfection unit offline for maintenance during that window.", "supports": [1], "contradicts": [0, 4], "base_confidence": 0.80},
            {"text": "Genomic sequencing (Oxford Nanopore, MinION): Nasopharyngeal isolates from 12 cases show H5N1 with PB2 E627K mutation (mammalian adaptation marker). HA cleavage site: polybasic (PLREKRRKR/GLF). Phylogenetic analysis places strain in clade 2.3.4.4b with novel reassortment of NA segment.", "supports": [2], "contradicts": [0, 1, 4], "base_confidence": 0.88},
            {"text": "Entomological survey (vector control district): Breteau Index 42 (high risk >20). Aedes aegypti trap positivity rate 67% in affected neighborhoods. Dengue NS1 antigen rapid test positive in 28 of 45 febrile cases. IgM/IgG serology pattern consistent with primary infection.", "supports": [3], "contradicts": [0, 1], "base_confidence": 0.78},
            {"text": "Environmental sampling (food safety lab): Pulsed-field gel electrophoresis (PFGE) pattern of Salmonella isolate from chicken processing plant matches clinical isolates (100% band match, 3 enzymes). Whole-genome SNP analysis: 2-SNP cluster across 8 states. Traceback to Farm ID 447-GA.", "supports": [0], "contradicts": [1, 2, 3, 4], "base_confidence": 0.90},
            {"text": "Epidemiological curve analysis: Incubation period distribution (median 31hr, range 18-56hr) is consistent with Norovirus (24-48hr typical). Secondary attack rate in households: 32%. Propagated pattern with 3 successive waves at 48-72hr intervals.", "supports": [4], "contradicts": [0, 3], "base_confidence": 0.72},
            {"text": "Geographic clustering (SaTScan analysis): Cases NOT confined to any single water district -- crosses 3 municipal water systems. Spatial cluster centroid is at a shopping mall food court. Relative risk within 2km radius: 5.7 (p=0.002).", "supports": [0, 4], "contradicts": [1], "base_confidence": 0.75},
            {"text": "Clinical presentation summary (n=189 confirmed cases): Fever >38.5C (89%), cough (76%), dyspnea (34%), bilateral infiltrates on CXR (28%). Hospitalization rate 18%. ICU admission 4.2%. CFR 1.1% (2 deaths, both >70yr with comorbidities). Median age: 41yr.", "supports": [2], "contradicts": [0, 4], "base_confidence": 0.70},
            {"text": "Contact tracing data (local health dept): Secondary cases cluster tightly around index cases -- no sustained community transmission. All secondary cases had prolonged face-to-face contact (>15 min, <2m) or shared contaminated surfaces. R0 estimated at 1.4 (95% CI: 0.9-2.1).", "supports": [4], "contradicts": [2, 3], "base_confidence": 0.68},
            {"text": "Travel and exposure history (line list analysis): 78% of cases report no travel. No cases among tourists or recent arrivals. Onset dates span 18 days. No common outdoor exposure. 62% of cases are food service workers or their household contacts.", "supports": [0, 4], "contradicts": [2, 3], "base_confidence": 0.65},
        ],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Internal data structures (not exposed to agents)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HiddenEvidence:
    """Server-side ground truth for one evidence item."""
    evidence_id: int
    text: str
    true_support: List[int]       # Hypothesis IDs it genuinely supports
    true_contradiction: List[int] # Hypothesis IDs it genuinely contradicts
    reliability: float            # 0.5–1.0; low → support may flip
    base_confidence: float        # How confident the source appears


@dataclass
class EpisodeData:
    """All hidden state for one episode."""
    seed: int
    theme: str
    domain: str
    true_hypothesis_id: int
    hypotheses: List[str]
    evidence: List[HiddenEvidence]
    active_hypothesis_ids: List[int] = field(default_factory=list)
    revealed_evidence: Dict[int, EvidenceItem] = field(default_factory=dict)
    expert_hints: List[ExpertHint] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    budget_remaining: int = 15
    total_budget: int = 15
    step_count: int = 0
    discarded_true: bool = False
    correct_discards: int = 0
    incorrect_discards: int = 0
    evidence_read: int = 0
    experiments_run: int = 0
    experts_consulted: int = 0
    raw_reward: float = 0.0
    submitted: bool = False
    submitted_hypothesis: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════════
# Action costs
# ═══════════════════════════════════════════════════════════════════════════

ACTION_COSTS: Dict[str, int] = {
    "read_evidence": 1,
    "run_experiment": 3,
    "consult_expert": 2,
    "discard_hypothesis": 0,
    "submit_answer": 0,
}

VALID_ACTION_TYPES = set(ACTION_COSTS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# The Environment
# ═══════════════════════════════════════════════════════════════════════════

class RCTDEnvironment(Environment[RCTDAction, RCTDObservation, RCTDState]):
    """Research Coordination & Truth Discovery Environment.

    Inherits from openenv.core.Environment for full framework compliance.
    Evaluates agent reasoning under conflicting, noisy evidence.

    Key mechanics:
      - Evidence may be noisy (support flips based on reliability).
      - ``run_experiment`` bypasses noise to reveal ground truth.
      - ``consult_expert`` provides probabilistic hints.
      - Budget constrains total actions; agents must be efficient.
      - Rich terminal metrics enable fine-grained evaluation.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._episode: Optional[EpisodeData] = None
        self._state = RCTDState()
        self._episode_id: Optional[str] = None

    def get_metadata(self):
        """Return environment metadata for the OpenEnv framework."""
        from openenv.core.env_server.interfaces import EnvironmentMetadata
        return EnvironmentMetadata(
            name="RCTD Environment",
            description="Research Coordination & Truth Discovery — "
                        "Epistemic reasoning under uncertainty",
            version="1.0.0",
        )

    # ───────────────────────────────────────────────────────────────────
    # reset()
    # ───────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "medium",
        **kwargs: Any,
    ) -> RCTDObservation:
        """Start a new episode. Deterministic given seed + task_id."""
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        self._episode_id = episode_id or str(uuid.uuid4())

        # Get task configuration
        config = TASK_CONFIGS.get(task_id, TASK_CONFIGS["medium"])

        # Generate the episode
        self._episode = _generate_episode(
            seed=seed,
            num_hypotheses=config["num_hypotheses"],
            num_evidence=config["num_evidence"],
            noise_level=config["noise_level"],
            budget=config["budget"],
        )

        # Update state
        self._state = RCTDState(
            episode_id=self._episode_id,
            step_count=0,
            task_id=task_id,
            seed=seed,
            num_hypotheses=config["num_hypotheses"],
            num_evidence=config["num_evidence"],
            noise_level=config["noise_level"],
            total_budget=config["budget"],
            true_hypothesis_id=self._episode.true_hypothesis_id,
            scenario_theme=self._episode.theme,
        )

        return RCTDObservation(
            done=False,
            reward=None,
            hypotheses=self._episode.hypotheses,
            active_hypothesis_ids=list(self._episode.active_hypothesis_ids),
            revealed_evidence=[],
            expert_hints=[],
            budget_remaining=self._episode.budget_remaining,
            total_evidence_count=len(self._episode.evidence),
            step_count=0,
            action_history=[],
            message=(
                f"Welcome to the {self._episode.domain} investigation. "
                f"There are {len(self._episode.hypotheses)} competing hypotheses "
                f"and {len(self._episode.evidence)} evidence items to investigate. "
                f"You have a budget of {self._episode.budget_remaining} action points. "
                f"Use your budget wisely to determine the truth."
            ),
            metrics=None,
        )

    # ───────────────────────────────────────────────────────────────────
    # step()
    # ───────────────────────────────────────────────────────────────────

    def step(
        self,
        action: RCTDAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> RCTDObservation:
        """Process one agent action. Returns updated observation."""
        ep = self._episode
        if ep is None:
            raise RuntimeError("Must call reset() before step()")

        if ep.submitted:
            raise RuntimeError("Episode already terminated. Call reset().")

        # Validate action type
        if action.type not in VALID_ACTION_TYPES:
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid action type '{action.type}'. "
                        f"Valid types: {sorted(VALID_ACTION_TYPES)}",
            )

        # Check budget
        cost = ACTION_COSTS[action.type]
        if cost > ep.budget_remaining and action.type != "submit_answer":
            # Force submission when budget is exhausted
            return self._force_budget_exhausted()

        ep.step_count += 1
        self._state.step_count = ep.step_count

        # Dispatch to action handler
        handler = {
            "read_evidence": self._handle_read_evidence,
            "run_experiment": self._handle_run_experiment,
            "consult_expert": self._handle_consult_expert,
            "discard_hypothesis": self._handle_discard_hypothesis,
            "submit_answer": self._handle_submit_answer,
        }[action.type]

        return handler(action)

    # ───────────────────────────────────────────────────────────────────
    # state property
    # ───────────────────────────────────────────────────────────────────

    @property
    def state(self) -> RCTDState:
        """Return current episode metadata (includes hidden ground truth)."""
        if self._episode:
            self._state.correct_discards = self._episode.correct_discards
            self._state.incorrect_discards = self._episode.incorrect_discards
            self._state.evidence_read = self._episode.evidence_read
            self._state.experiments_run = self._episode.experiments_run
            self._state.experts_consulted = self._episode.experts_consulted
            self._state.raw_reward = self._episode.raw_reward
        return self._state

    # ───────────────────────────────────────────────────────────────────
    # Action Handlers
    # ───────────────────────────────────────────────────────────────────

    def _handle_read_evidence(self, action: RCTDAction) -> RCTDObservation:
        ep = self._episode
        eid = action.evidence_id

        if eid is None or eid < 0 or eid >= len(ep.evidence):
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid evidence_id={eid}. "
                        f"Valid range: 0–{len(ep.evidence) - 1}",
            )

        if eid in ep.revealed_evidence:
            return self._make_obs(
                reward=0.0,
                message=f"Evidence E{eid} has already been read.",
            )

        # Deduct cost
        ep.budget_remaining -= ACTION_COSTS["read_evidence"]
        ep.evidence_read += 1

        # Get the hidden evidence
        hidden = ep.evidence[eid]

        # Apply noise: low reliability → support may flip
        rng = random.Random(ep.seed * 1000 + eid)
        apparent_support = list(hidden.true_support)
        apparent_contradiction = list(hidden.true_contradiction)

        if rng.random() > hidden.reliability:
            # Noise: flip support — remove some true supports, add false ones
            if apparent_support and rng.random() < 0.5:
                apparent_support.pop(rng.randrange(len(apparent_support)))
            # Add a false support
            false_candidates = [
                h for h in ep.active_hypothesis_ids
                if h not in hidden.true_support and h != ep.true_hypothesis_id
            ]
            if false_candidates:
                apparent_support.append(rng.choice(false_candidates))
            # Noise on contradictions too: may drop a real contradiction
            if apparent_contradiction and rng.random() < 0.4:
                apparent_contradiction.pop(rng.randrange(len(apparent_contradiction)))

        item = EvidenceItem(
            evidence_id=eid,
            text=hidden.text,
            apparent_support=apparent_support,
            apparent_contradiction=apparent_contradiction,
            confidence=hidden.base_confidence,
            verified=False,
        )
        ep.revealed_evidence[eid] = item

        # Step reward proportional to action cost
        step_reward = -float(ACTION_COSTS['read_evidence'])
        ep.raw_reward += step_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "read_evidence",
            "evidence_id": eid,
            "cost": ACTION_COSTS["read_evidence"],
        })

        # Check if budget exhausted
        if ep.budget_remaining <= 0:
            return self._force_budget_exhausted()

        return self._make_obs(
            reward=self._normalize_step_reward(step_reward),
            message=f"Read evidence E{eid}: \"{hidden.text}\" "
                    f"— Appears to support hypothesis/hypotheses: "
                    f"{[f'H{s}' for s in apparent_support]} "
                    f"(confidence: {hidden.base_confidence:.0%})",
        )

    def _handle_run_experiment(self, action: RCTDAction) -> RCTDObservation:
        ep = self._episode
        eid = action.evidence_id

        if eid is None or eid < 0 or eid >= len(ep.evidence):
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid evidence_id={eid}. "
                        f"Valid range: 0–{len(ep.evidence) - 1}",
            )

        if eid not in ep.revealed_evidence:
            return self._make_obs(
                reward=-0.01,
                message=f"Must read evidence E{eid} first before running experiment.",
            )

        if ep.revealed_evidence[eid].verified:
            return self._make_obs(
                reward=0.0,
                message=f"Evidence E{eid} has already been verified.",
            )

        # Deduct cost
        ep.budget_remaining -= ACTION_COSTS["run_experiment"]
        ep.experiments_run += 1

        # Reveal ground truth (strips noise)
        hidden = ep.evidence[eid]
        verified_item = EvidenceItem(
            evidence_id=eid,
            text=hidden.text,
            apparent_support=list(hidden.true_support),
            apparent_contradiction=list(hidden.true_contradiction),
            confidence=1.0,  # Verified = full confidence
            verified=True,
        )
        ep.revealed_evidence[eid] = verified_item

        # Step reward proportional to action cost
        step_reward = -float(ACTION_COSTS['run_experiment'])
        ep.raw_reward += step_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "run_experiment",
            "evidence_id": eid,
            "cost": ACTION_COSTS["run_experiment"],
        })

        if ep.budget_remaining <= 0:
            return self._force_budget_exhausted()

        return self._make_obs(
            reward=self._normalize_step_reward(step_reward),
            message=f"Experiment on E{eid} complete. VERIFIED support: "
                    f"{[f'H{s}' for s in hidden.true_support]} "
                    f"(confidence: 100%)",
        )

    def _handle_consult_expert(self, action: RCTDAction) -> RCTDObservation:
        ep = self._episode
        hid = action.hypothesis_id

        if hid is None or hid not in ep.active_hypothesis_ids:
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid or already-discarded hypothesis_id={hid}.",
            )

        # Deduct cost
        ep.budget_remaining -= ACTION_COSTS["consult_expert"]
        ep.experts_consulted += 1

        # Generate probabilistic hint (deterministic from seed)
        rng = random.Random(ep.seed * 2000 + hid)

        if hid == ep.true_hypothesis_id:
            # True hypothesis: expert gives high probability (0.6–0.9)
            prob = rng.uniform(0.6, 0.9)
            hint_text = _generate_expert_hint_text(ep.theme, hid, prob, is_true=True, rng=rng)
        else:
            # False hypothesis: expert gives low probability (0.1–0.45)
            prob = rng.uniform(0.1, 0.45)
            hint_text = _generate_expert_hint_text(ep.theme, hid, prob, is_true=False, rng=rng)

        hint = ExpertHint(
            hypothesis_id=hid,
            hint_text=hint_text,
            estimated_probability=round(prob, 2),
        )
        ep.expert_hints.append(hint)

        step_reward = -float(ACTION_COSTS['consult_expert'])
        ep.raw_reward += step_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "consult_expert",
            "hypothesis_id": hid,
            "cost": ACTION_COSTS["consult_expert"],
        })

        if ep.budget_remaining <= 0:
            return self._force_budget_exhausted()

        return self._make_obs(
            reward=self._normalize_step_reward(step_reward),
            message=f"Expert consulted on H{hid}: \"{hint_text}\" "
                    f"(estimated probability: {prob:.0%})",
        )

    def _handle_discard_hypothesis(self, action: RCTDAction) -> RCTDObservation:
        ep = self._episode
        hid = action.hypothesis_id

        if hid is None or hid not in ep.active_hypothesis_ids:
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid or already-discarded hypothesis_id={hid}.",
            )

        # Cannot discard the last remaining hypothesis
        if len(ep.active_hypothesis_ids) <= 1:
            return self._make_obs(
                reward=-0.01,
                message="Cannot discard the last remaining hypothesis. "
                        "Use submit_answer instead.",
            )

        ep.active_hypothesis_ids.remove(hid)

        if hid == ep.true_hypothesis_id:
            # Agent discarded the truth — severe penalty
            ep.discarded_true = True
            ep.incorrect_discards += 1
            step_reward = -5.0
            msg = f"Discarded hypothesis H{hid}: \"{ep.hypotheses[hid]}\". Noted."
        else:
            # Correct discard
            ep.correct_discards += 1
            step_reward = 2.0
            msg = f"Discarded hypothesis H{hid}: \"{ep.hypotheses[hid]}\". " \
                  f"Search space narrowed to {len(ep.active_hypothesis_ids)} hypotheses."

        ep.raw_reward += step_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "discard_hypothesis",
            "hypothesis_id": hid,
            "cost": 0,
        })

        return self._make_obs(
            reward=self._normalize_step_reward(step_reward),
            message=msg,
        )

    def _handle_submit_answer(self, action: RCTDAction) -> RCTDObservation:
        ep = self._episode
        hid = action.hypothesis_id

        if hid is None or hid < 0 or hid >= len(ep.hypotheses):
            return self._make_obs(
                reward=-0.01,
                message=f"Invalid hypothesis_id={hid} for submission. "
                        f"Valid: {list(range(len(ep.hypotheses)))}",
            )

        ep.submitted = True
        ep.submitted_hypothesis = hid

        # Terminal reward
        correct = (hid == ep.true_hypothesis_id)

        if correct and ep.budget_remaining > 0:
            terminal_reward = 100.0
        elif correct and ep.budget_remaining == 0:
            terminal_reward = 50.0
        elif not correct and ep.discarded_true:
            terminal_reward = -100.0
        else:
            terminal_reward = -50.0

        ep.raw_reward += terminal_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "submit_answer",
            "hypothesis_id": hid,
            "cost": 0,
            "correct": correct,
        })

        # Compute terminal metrics
        metrics = _compute_metrics(ep)

        if correct:
            msg = (f"✓ CORRECT! The answer is H{hid}: \"{ep.hypotheses[hid]}\". "
                   f"Efficiency: {metrics['efficiency_score']:.0%}")
        else:
            msg = (f"✗ INCORRECT. You answered H{hid}: \"{ep.hypotheses[hid]}\". "
                   f"The true answer was H{ep.true_hypothesis_id}: "
                   f"\"{ep.hypotheses[ep.true_hypothesis_id]}\".")

        # Normalize total raw reward to 0.0–1.0 for the observation
        normalized = self._normalize_terminal_reward(ep.raw_reward)

        return RCTDObservation(
            done=True,
            reward=normalized,
            hypotheses=ep.hypotheses,
            active_hypothesis_ids=list(ep.active_hypothesis_ids),
            revealed_evidence=list(ep.revealed_evidence.values()),
            expert_hints=list(ep.expert_hints),
            budget_remaining=ep.budget_remaining,
            total_evidence_count=len(ep.evidence),
            step_count=ep.step_count,
            action_history=ep.action_history,
            message=msg,
            metrics=metrics,
        )

    # ───────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────

    def _force_budget_exhausted(self) -> RCTDObservation:
        """Budget hit zero — force termination.

        Unlike voluntary submission, budget exhaustion picks the
        most-supported hypothesis from gathered evidence. This is
        intentionally weaker than the agent choosing deliberately
        (the agent is penalized for running out of budget through
        reduced efficiency score and the budget_exhausted failure mode).
        """
        ep = self._episode
        ep.submitted = True

        # Blindly use the first active hypothesis as a fallback.
        # We no longer use a smart heuristic fallback because it artificially
        # inflates the scores of random agents that exhaust their budget.
        if ep.active_hypothesis_ids:
            auto_answer = ep.active_hypothesis_ids[0]
        else:
            auto_answer = 0

        ep.submitted_hypothesis = auto_answer
        correct = (auto_answer == ep.true_hypothesis_id)

        terminal_reward = -50.0 if not correct else 50.0
        ep.raw_reward += terminal_reward

        ep.action_history.append({
            "step": ep.step_count,
            "action": "budget_exhausted",
            "auto_answer": auto_answer,
            "correct": correct,
        })

        metrics = _compute_metrics(ep)
        metrics["failure_mode"] = "budget_exhausted"

        normalized = self._normalize_terminal_reward(ep.raw_reward)

        msg = (
            f"⚠ Budget exhausted! Auto-submitted H{auto_answer}. "
            + ("Correct!" if correct else
               f"Wrong — true answer was H{ep.true_hypothesis_id}.")
        )

        return RCTDObservation(
            done=True,
            reward=normalized,
            hypotheses=ep.hypotheses,
            active_hypothesis_ids=list(ep.active_hypothesis_ids),
            revealed_evidence=list(ep.revealed_evidence.values()),
            expert_hints=list(ep.expert_hints),
            budget_remaining=0,
            total_evidence_count=len(ep.evidence),
            step_count=ep.step_count,
            action_history=ep.action_history,
            message=msg,
            metrics=metrics,
        )

    def _make_obs(self, reward: float, message: str) -> RCTDObservation:
        """Build an observation from current episode state."""
        ep = self._episode
        return RCTDObservation(
            done=False,
            reward=self._normalize_step_reward(reward),
            hypotheses=ep.hypotheses,
            active_hypothesis_ids=list(ep.active_hypothesis_ids),
            revealed_evidence=list(ep.revealed_evidence.values()),
            expert_hints=list(ep.expert_hints),
            budget_remaining=ep.budget_remaining,
            total_evidence_count=len(ep.evidence),
            step_count=ep.step_count,
            action_history=ep.action_history,
            message=message,
            metrics=None,
        )

    _EPS = 1e-4  # Strict (0, 1) — validator rejects exact 0.0 and 1.0

    @staticmethod
    def _normalize_step_reward(raw: float) -> float:
        """Map step rewards to (0, 1) via linear clamp.

        Raw step rewards range from approx -5 (worst) to +2 (best).
        Linear mapping avoids the sigmoid issue where raw=0 → 0.5.
        Clamped to open interval (ε, 1−ε) per OpenEnv validation spec.
        """
        # Linear: map [-5, +2] → (0, 1)
        normalized = (raw + 5.0) / 7.0
        return round(max(RCTDEnvironment._EPS, min(1.0 - RCTDEnvironment._EPS, normalized)), 4)

    @staticmethod
    def _normalize_terminal_reward(raw: float) -> float:
        """Map cumulative raw reward to (0, 1).

        Raw range: approx -120 (worst) to +120 (best).
        Clamped to open interval (ε, 1−ε) per OpenEnv validation spec.
        """
        # Sigmoid normalization centered at 0
        normalized = 1.0 / (1.0 + math.exp(-raw / 30.0))
        return round(max(RCTDEnvironment._EPS, min(1.0 - RCTDEnvironment._EPS, normalized)), 4)


# ═══════════════════════════════════════════════════════════════════════════
# Task Configurations
# ═══════════════════════════════════════════════════════════════════════════

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "num_hypotheses": 3,
        "num_evidence": 6,
        "noise_level": 0.1,
        "budget": 20,
        "description": "Introductory difficulty — low noise, generous budget",
    },
    "medium": {
        "num_hypotheses": 4,
        "num_evidence": 8,
        "noise_level": 0.3,
        "budget": 12,
        "description": "Standard challenge — moderate noise, tighter budget",
    },
    "hard": {
        "num_hypotheses": 5,
        "num_evidence": 10,
        "noise_level": 0.50,
        "budget": 8,
        "description": "Expert difficulty — high noise, tight budget, deep reasoning required",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Expert Hint Generation (Domain-Specific, Graduated)
# ═══════════════════════════════════════════════════════════════════════════

# Per-domain expert hint templates. Each list has 3 entries:
#   [0] = high confidence (prob > 0.7)
#   [1] = moderate confidence (prob 0.4–0.7)
#   [2] = low confidence (prob < 0.4)
_EXPERT_HINT_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "medical_diagnosis": {
        "positive": [
            "The clinical presentation, combined with lab markers, strongly aligns with H{hid}. I've seen this pattern repeatedly in my practice.",
            "Several diagnostic indicators are consistent with H{hid}, though I'd recommend confirmatory testing before concluding.",
            "There are some features suggestive of H{hid}, but the differential is still broad at this stage.",
        ],
        "negative": [
            "The symptom profile has critical inconsistencies with H{hid} — multiple pathognomonic signs are absent.",
            "While H{hid} was initially plausible, the accumulating evidence is making me increasingly skeptical.",
            "H{hid} cannot be ruled out entirely, but other etiologies seem more consistent with the observed data.",
        ],
    },
    "market_analysis": {
        "positive": [
            "The macroeconomic signals and sector-specific data strongly corroborate H{hid}. My quantitative models converge on this explanation.",
            "H{hid} is consistent with several leading indicators I track, though the causal link isn't conclusive yet.",
            "H{hid} has some supporting data points, but the market dynamics are complex enough that I'd want more confirmation.",
        ],
        "negative": [
            "Key market fundamentals directly contradict H{hid} — the correlations break down under scrutiny.",
            "My models show diminishing support for H{hid} as more data comes in. The trend is moving away from this thesis.",
            "H{hid} is one possible factor, but it's unlikely to be the primary driver based on what I'm seeing.",
        ],
    },
    "security_incident": {
        "positive": [
            "The attack signature, TTPs, and lateral movement pattern are highly characteristic of H{hid}. I've investigated similar incidents before.",
            "Several forensic indicators point toward H{hid}, though we need to complete the full kill-chain analysis.",
            "H{hid} is among the plausible scenarios, but the evidence is still circumstantial at this point.",
        ],
        "negative": [
            "The IOCs and forensic timeline have significant gaps that are inconsistent with H{hid}.",
            "Some aspects of H{hid} don't match the attack surface analysis. I'm leaning away from this theory.",
            "H{hid} is possible but less likely given the IR data collected so far.",
        ],
    },
    "climate_attribution": {
        "positive": [
            "The spatial and temporal patterns in the observational data are strongly consistent with H{hid} — multiple independent datasets converge.",
            "Modeling results and proxy data offer moderate support for H{hid}, though natural variability adds uncertainty.",
            "H{hid} is within the range of plausible forcings, but the attribution signal is not yet robust.",
        ],
        "negative": [
            "The observational record shows patterns that are fundamentally inconsistent with H{hid} as the primary driver.",
            "Recent paleoclimate comparisons weaken the case for H{hid}. The forcing magnitude doesn't match.",
            "H{hid} likely contributes but is unlikely to be the dominant factor based on current analysis.",
        ],
    },
    "artifact_authentication": {
        "positive": [
            "The material composition, patina development, and stylistic elements are all strongly consistent with H{hid}. This is a compelling case.",
            "Several analytical results support H{hid}, but I'd want to see additional provenance documentation.",
            "H{hid} remains plausible based on initial examination, though the evidence is not yet definitive.",
        ],
        "negative": [
            "Critical anachronisms and material inconsistencies make H{hid} very unlikely in my professional assessment.",
            "The isotopic analysis and tool-mark patterns are creating doubt about H{hid}.",
            "H{hid} cannot be excluded, but it doesn't fit the overall pattern of evidence as cleanly as alternatives.",
        ],
    },
    "legal_dispute": {
        "positive": [
            "The documentary evidence, witness testimony, and financial records form a compelling narrative supporting H{hid}. In my experience litigating similar cases, this pattern is highly probative.",
            "Several key exhibits are consistent with H{hid}, though opposing counsel will likely challenge the chain of custody on some documents.",
            "H{hid} has some supporting indicators, but the evidentiary standard for this claim type is high. More discovery may be needed.",
        ],
        "negative": [
            "The documentary record has critical gaps that undermine H{hid}. The timeline inconsistencies alone would be fatal at trial.",
            "While H{hid} was initially plausible, the deposition testimony and financial forensics are moving the needle away from this theory.",
            "H{hid} remains in the differential, but the burden of proof considerations make it a weaker position than alternatives.",
        ],
    },
    "outbreak_investigation": {
        "positive": [
            "The epidemiological curve, case clustering, and laboratory confirmation strongly support H{hid}. I've investigated similar outbreaks at CDC and this signature is characteristic.",
            "The attack rate data and exposure history are moderately consistent with H{hid}, though we're still awaiting confirmatory lab results.",
            "H{hid} is among the plausible transmission routes, but the case-control study needs more statistical power before we can be confident.",
        ],
        "negative": [
            "The incubation period distribution and spatial clustering are fundamentally inconsistent with H{hid}. The biology doesn't support this transmission route.",
            "Accumulating lab data and contact tracing results are making H{hid} less likely. The secondary attack rate pattern doesn't fit.",
            "H{hid} cannot be fully excluded, but the epidemiological evidence is pointing toward a different source.",
        ],
    },
}


def _generate_expert_hint_text(
    theme: str,
    hid: int,
    prob: float,
    is_true: bool,
    rng: random.Random,
) -> str:
    """Generate a domain-specific expert hint with graduated confidence."""
    templates = _EXPERT_HINT_TEMPLATES.get(theme)

    if templates is None:
        # Fallback for unknown themes
        if is_true:
            return f"Based on my expertise, hypothesis H{hid} shows indicative evidence in its favor."
        return f"Based on my expertise, hypothesis H{hid} has aspects that raise questions."

    key = "positive" if is_true else "negative"
    variants = templates[key]

    # Select based on probability level
    if prob > 0.7:
        hint = variants[0]
    elif prob > 0.4:
        hint = variants[1]
    else:
        hint = variants[2]

    return hint.format(hid=hid)


# ═══════════════════════════════════════════════════════════════════════════
# Evidence-Based Hypothesis Selection
# ═══════════════════════════════════════════════════════════════════════════


def _find_best_hypothesis_from_evidence(
    active_hypothesis_ids: List[int],
    revealed_evidence: Dict[int, EvidenceItem],
    expert_hints: List[ExpertHint],
) -> int:
    """Pick the most-supported active hypothesis from gathered evidence.

    Used by _force_budget_exhausted() to make a best-effort answer
    when the agent runs out of budget. Weights verified evidence
    higher than unverified, and factors in expert opinions.
    """
    if not active_hypothesis_ids:
        return 0

    support_counts: Dict[int, float] = {h: 0.0 for h in active_hypothesis_ids}

    # Count evidence support (verified evidence gets double weight)
    for ev in revealed_evidence.values():
        weight = 2.0 if ev.verified else 1.0
        for h in ev.apparent_support:
            if h in support_counts:
                support_counts[h] += weight * ev.confidence

    # Factor in expert hints
    for hint in expert_hints:
        if hint.hypothesis_id in support_counts:
            support_counts[hint.hypothesis_id] += hint.estimated_probability

    return max(support_counts, key=support_counts.get)


# ═══════════════════════════════════════════════════════════════════════════
# Episode Generation (Deterministic Epistemic Engine)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_episode(
    seed: int,
    num_hypotheses: int,
    num_evidence: int,
    noise_level: float,
    budget: int,
) -> EpisodeData:
    """Generate a complete episode from seed. STRICTLY DETERMINISTIC.

    Evidence rules:
      - 60–70% of evidence implicitly supports the true hypothesis
      - 1–2 high-confidence misleading contradictions
      - At least 2 strong, indisputable support items for truth
      - False hypotheses get 20–40% accidental support
      - Reliability varies: low reliability → noise flips support
    """
    rng = random.Random(seed)

    # Select scenario
    template = rng.choice(SCENARIO_TEMPLATES)

    # Select hypotheses (subset of pool)
    assert num_hypotheses <= len(template.hypothesis_pool), \
        f"Requested {num_hypotheses} hypotheses but template only has {len(template.hypothesis_pool)}"
    hypothesis_indices = rng.sample(range(len(template.hypothesis_pool)), num_hypotheses)
    hypotheses = [template.hypothesis_pool[i] for i in hypothesis_indices]

    # Remap indices: original template IDs → new 0..N-1 IDs
    index_remap = {old: new for new, old in enumerate(hypothesis_indices)}

    # Select ground truth
    true_id = rng.randrange(num_hypotheses)

    # Select and adapt evidence
    available_evidence = list(range(len(template.evidence_templates)))
    rng.shuffle(available_evidence)
    selected_evidence_indices = available_evidence[:num_evidence]

    evidence_items: List[HiddenEvidence] = []
    strong_support_count = 0

    for new_eid, orig_eid in enumerate(selected_evidence_indices):
        tmpl = template.evidence_templates[orig_eid]

        # Remap support/contradiction to our hypothesis indices
        true_support = [
            index_remap[s] for s in tmpl["supports"]
            if s in index_remap
        ]
        true_contradiction = [
            index_remap[c] for c in tmpl["contradicts"]
            if c in index_remap
        ]

        # Ensure evidence distribution rules
        # If this evidence doesn't yet support truth and we need more support:
        if true_id not in true_support and rng.random() < 0.65:
            true_support.append(true_id)

        # Determine reliability based on noise level
        reliability = rng.uniform(max(0.5, 1.0 - noise_level * 1.5), 1.0)

        # Create at least 2 strong support items for the true hypothesis
        base_confidence = tmpl["base_confidence"]
        if true_id in true_support and strong_support_count < 2:
            reliability = max(reliability, 0.85)
            base_confidence = max(base_confidence, 0.8)
            strong_support_count += 1

        # Create 1-2 misleading high-confidence contradictions
        if (new_eid < 2 and true_id in true_support
                and rng.random() < noise_level):
            # Make this a misleading item
            reliability = rng.uniform(0.4, 0.6)
            base_confidence = rng.uniform(0.7, 0.9)  # Looks confident but unreliable

        evidence_items.append(HiddenEvidence(
            evidence_id=new_eid,
            text=tmpl["text"],
            true_support=true_support,
            true_contradiction=true_contradiction,
            reliability=round(reliability, 3),
            base_confidence=round(base_confidence, 2),
        ))

    # Ensure false hypotheses have 20-40% accidental support
    for h_id in range(num_hypotheses):
        if h_id == true_id:
            continue
        supporting = sum(1 for e in evidence_items if h_id in e.true_support)
        target_support = rng.randint(
            max(1, int(num_evidence * 0.2)),
            max(2, int(num_evidence * 0.4)),
        )
        while supporting < target_support:
            candidate = rng.choice(evidence_items)
            if h_id not in candidate.true_support:
                candidate.true_support.append(h_id)
                supporting += 1

    return EpisodeData(
        seed=seed,
        theme=template.theme,
        domain=template.domain,
        true_hypothesis_id=true_id,
        hypotheses=hypotheses,
        evidence=evidence_items,
        active_hypothesis_ids=list(range(num_hypotheses)),
        budget_remaining=budget,
        total_budget=budget,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Terminal Metrics
# ═══════════════════════════════════════════════════════════════════════════

def _compute_metrics(ep: EpisodeData) -> Dict[str, Any]:
    """Compute rich evaluation metrics for the info dict."""
    total_evidence = len(ep.evidence)
    revealed = len(ep.revealed_evidence)

    # Success
    success = (ep.submitted_hypothesis == ep.true_hypothesis_id)

    # Efficiency: budget remaining / total budget
    efficiency = ep.budget_remaining / ep.total_budget if ep.total_budget > 0 else 0

    # Evidence utilization
    utilization = revealed / total_evidence if total_evidence > 0 else 0

    # Failure mode classification
    failure_mode = None
    if not success:
        if ep.discarded_true:
            failure_mode = "discarded_correct_hypothesis"
        elif ep.budget_remaining <= 0:
            failure_mode = "budget_exhausted"
        elif revealed < total_evidence * 0.3:
            failure_mode = "insufficient_evidence"
        else:
            # Check if agent was misled by noisy evidence
            noisy_reads = sum(
                1 for eid, item in ep.revealed_evidence.items()
                if not item.verified and ep.evidence[eid].reliability < 0.7
            )
            if noisy_reads > 0:
                failure_mode = "misled_by_noise"
            else:
                failure_mode = "reasoning_error"

    return {
        "success": success,
        "efficiency_score": round(efficiency, 3),
        "evidence_utilization": round(utilization, 3),
        "steps_taken": ep.step_count,
        "evidence_read": ep.evidence_read,
        "experiments_run": ep.experiments_run,
        "experts_consulted": ep.experts_consulted,
        "correct_discards": ep.correct_discards,
        "incorrect_discards": ep.incorrect_discards,
        "budget_used": ep.total_budget - ep.budget_remaining,
        "failure_mode": failure_mode,
        "raw_reward": round(ep.raw_reward, 2),
        "scenario_theme": ep.theme,
        "true_hypothesis": ep.true_hypothesis_id,
        "submitted_hypothesis": ep.submitted_hypothesis,
    }
