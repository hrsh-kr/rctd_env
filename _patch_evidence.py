#!/usr/bin/env python3
"""Patch: Add 2 new domains, add evidence to existing domains, fix hard difficulty."""
import re

# Read file
with open('rctd_env/server/environment.py', 'r') as f:
    content = f.read()

# ============================================================
# 1. Add 2 new domains + 5 extra evidence per existing domain
# ============================================================

# New domains to add before the closing ']'
NEW_DOMAINS = r'''    # -- 6. Legal Contract Dispute -----------------------------------------
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
'''

# Extra evidence for existing domains (5 per domain)
EXTRA_MEDICAL = r'''            {"text": "Blood smear peripheral morphology: Toxic granulation and Dohle bodies in neutrophils. No schistocytes or spherocytes. Reticulocyte count 1.8% (normal). Haptoglobin 145 mg/dL (normal), ruling out hemolytic process.", "supports": [0], "contradicts": [3, 4], "base_confidence": 0.72},
            {"text": "Lumbar puncture results: CSF WBC 2 cells/uL (normal), protein 38 mg/dL (normal), glucose 62 mg/dL (normal). No organisms on Gram stain. CSF HSV PCR negative. Opening pressure 14 cmH2O.", "supports": [0, 3], "contradicts": [1], "base_confidence": 0.60},
            {"text": "Urine toxicology screen (GC-MS): Positive for organophosphate metabolites (diethylphosphate 142 ug/L, diethylthiophosphate 89 ug/L). Serum atropine challenge test: marked improvement in secretions within 5 minutes of 2mg IV atropine.", "supports": [2], "contradicts": [0, 1, 3, 4], "base_confidence": 0.87},
            {"text": "Echocardiogram (TTE): No vegetations on valves (Duke criteria not met). LVEF 58% (normal). No pericardial effusion. Aortic and mitral valves morphologically normal. Rules out infective endocarditis as source of bacteremia.", "supports": [1, 3], "contradicts": [0], "base_confidence": 0.58},
            {"text": "Skin biopsy (punch, 4mm, malar area): Interface dermatitis with vacuolar changes at DEJ. Mucin deposition in dermis. Direct immunofluorescence: granular IgG and C3 deposits at basement membrane zone (lupus band positive).", "supports": [3], "contradicts": [0, 1, 2, 4], "base_confidence": 0.86},
'''

EXTRA_MARKET = r'''            {"text": "Shipping satellite AIS data (MarineTraffic): 47 container vessels currently holding at Bab el-Mandab strait anchorage. Average wait time 8.4 days (normal: <1 day). CMA CGM and Hapag-Lloyd have rerouted 100% of Asia-Europe services via Cape of Good Hope since Jan 15.", "supports": [0], "contradicts": [1, 3, 4], "base_confidence": 0.83},
            {"text": "Commercial real estate REIT earnings (Q3): Boston Properties FFO down 14% YoY. Vornado Realty announced conversion of 3 office towers to residential. National office lease renewal rate: 61% (pre-pandemic: 82%). Sublease availability at 10-year high.", "supports": [1], "contradicts": [0], "base_confidence": 0.77},
            {"text": "SEC 8-K filing analysis: 23 EU-headquartered tech companies disclosed material AI Act compliance costs in recent filings. Average estimated spend: $45M for large-cap, $8M for mid-cap. Three companies announced relocation of AI research divisions to non-EU jurisdictions.", "supports": [2], "contradicts": [0, 4], "base_confidence": 0.72},
            {"text": "Semiconductor supply chain data (SEMI): Global wafer shipments up 12% QoQ. TSMC utilization rate at 94%. No reported shortages in consumer electronics or automotive chips. Lead times for standard components returned to pre-pandemic levels of 12-14 weeks.", "supports": [1, 2, 3, 4], "contradicts": [0], "base_confidence": 0.68},
            {"text": "Consumer confidence index (Conference Board): Present Situation Index fell to 134.5 (from 147.2). Expectations Index at 72.8 (recession signal <80). Labor differential (jobs plentiful minus hard to get) narrowed to 12.7 points, lowest since 2021.", "supports": [1, 4], "contradicts": [3], "base_confidence": 0.63},
'''

EXTRA_CYBER = r'''            {"text": "Active Directory audit log: Service account svc-backup had its password reset by J.M.'s admin credentials 72 hours before termination. Same service account was used for lateral movement to 3 database servers containing customer PII. No helpdesk ticket corresponds to this password reset.", "supports": [0], "contradicts": [1, 2, 4], "base_confidence": 0.82},
            {"text": "Mandiant threat intelligence report: APT29 shifted to cloud-focused operations in 2024. New tactic: abuse of trusted cloud services (Azure, AWS) for C2. No direct evidence of on-premise intrusion in this campaign. Targeting profile: government contractors and think tanks.", "supports": [1], "contradicts": [0, 4], "base_confidence": 0.70},
            {"text": "Patch management audit (Qualys): 67% of external-facing systems had critical patches applied within SLA. XZ Utils specifically was patched on 14 of 14 affected hosts within 48 hours of CVE publication. However, backdoor may have been active for up to 3 weeks before CVE disclosure.", "supports": [2], "contradicts": [], "base_confidence": 0.65},
            {"text": "Microsoft 365 audit log: Compromised OAuth tokens were used to create 3 new mail flow rules forwarding all C-suite email to external address. Rules created at 14:22 UTC using legitimate user session from residential ISP IP. Geo-impossible travel detected: London (14:00) -> Lagos (14:22).", "supports": [3], "contradicts": [0, 2], "base_confidence": 0.76},
            {"text": "AWS Config timeline: IAM policy arn:aws:iam::123456789:policy/S3FullAccess was attached to 8 additional roles between Jan 10-15. Change was made via Terraform by the deploy-automation role. No code review or approval in GitHub for corresponding IaC changes.", "supports": [4], "contradicts": [1, 2], "base_confidence": 0.79},
'''

EXTRA_CLIMATE = r'''            {"text": "Ocean buoy network (TAO/TRITON array): Subsurface temperature at 150m depth along equatorial Pacific shows +4.2C anomaly. Warm water volume (>20C isotherm) expanded to 135% of climatological mean. Thermocline depth anomaly: +28m. Pattern is textbook El Nino Modoki (central Pacific type).", "supports": [0], "contradicts": [3, 4], "base_confidence": 0.80},
            {"text": "Ice core proxy data (Vostok, EPICA): Current CO2 level (425 ppm) exceeds any value in the 800,000-year record. Rate of increase (3.4 ppm/yr) is 10x faster than the fastest natural increase during deglaciation events. Methane similarly unprecedented at 1,923 ppb.", "supports": [1], "contradicts": [], "base_confidence": 0.88},
            {"text": "SO2 monitoring (Aura/OMI satellite): No significant SO2 plumes detected in the past 6 months. Stratospheric SO2 column density at background levels (0.1 DU). No volcanic eruptions with VEI >= 4 since Hunga Tonga. Rules out fresh volcanic aerosol injection.", "supports": [0, 1, 4], "contradicts": [2], "base_confidence": 0.73},
            {"text": "Microclimate sensor network (urban/rural paired stations): Temperature differential exhibits strong diurnal cycle -- UHI effect peaks at 01:00-04:00 local (delta: +5.8C) and minimizes at 14:00 (delta: +1.2C). Weekend vs weekday difference: 0.4C (anthropogenic heat flux signature).", "supports": [3], "contradicts": [], "base_confidence": 0.62},
            {"text": "GRACE-FO satellite gravimetry: Terrestrial water storage in Amazon basin decreased by 34 km3/year over 2020-2024 period. Evapotranspiration reduction detected via MODIS ET product. Regional precipitation recycling ratio dropped from 0.35 to 0.28 (moisture recycling breakdown).", "supports": [4], "contradicts": [3], "base_confidence": 0.74},
'''

EXTRA_ARTIFACT = r'''            {"text": "Metallographic cross-section (optical microscopy, 200x): Alpha-phase bronze with equiaxed grain structure (avg grain size 45um) and annealing twins -- consistent with repeated hot-working and annealing cycles used in ancient lost-wax casting. No cold-rolled microstructure.", "supports": [0, 2], "contradicts": [1, 3], "base_confidence": 0.74},
            {"text": "Accelerator mass spectrometry (AMS) of core clay organic inclusions: Radiocarbon age 1,950 +/- 55 BP (calibrated: 20-130 CE). This is statistically inconsistent with the claimed 4th century BCE date at >99% confidence. Consistent with Roman Imperial period production.", "supports": [2], "contradicts": [0, 1, 3], "base_confidence": 0.86},
            {"text": "Portable X-ray diffraction (p-XRD) of blue pigment on surface: Egyptian Blue (CaCuSi2O6) identified. This pigment was used extensively from 3rd millennium BCE through 4th century CE. Its presence is consistent with both Hellenistic and Roman dates but rules out 19th century manufacture.", "supports": [0, 2], "contradicts": [1, 3], "base_confidence": 0.68},
            {"text": "Inscription analysis (Dr. A. Matthaiou, epigraphist): Letter forms show sigma with 4 bars (archaic) but omega with modern proportions (post-3rd century BCE). Mixed chronological indicators suggest either transitional period (late 4th century) or deliberate archaizing in Roman era.", "supports": [0, 2], "contradicts": [1, 3], "base_confidence": 0.58},
            {"text": "3D surface scanning (structured light, 50um accuracy): Left arm attachment surface shows fresh fracture faces without patination, surrounded by artificial patina (copper chloride solution application detected by FTIR). Attachment method uses modern stainless steel dowel pins (316L grade).", "supports": [4], "contradicts": [0, 2], "base_confidence": 0.84},
'''

# Insert extra evidence into each domain
# Medical
content = content.replace(
    '            {"text": "Patient history: 34F, no chemical exposure',
    EXTRA_MEDICAL + '            {"text": "Patient history: 34F, no chemical exposure'
)

# Market
content = content.replace(
    '            {"text": "Patent filing data (WIPO)',
    EXTRA_MARKET + '            {"text": "Patent filing data (WIPO)'
)

# Cyber
content = content.replace(
    '            {"text": "Network forensics: TLS certificate',
    EXTRA_CYBER + '            {"text": "Network forensics: TLS certificate'
)

# Climate
content = content.replace(
    '            {"text": "Flux tower network (FLUXNET)',
    EXTRA_CLIMATE + '            {"text": "Flux tower network (FLUXNET)'
)

# Artifact
content = content.replace(
    '            {"text": "Neutron activation analysis (NAA)',
    EXTRA_ARTIFACT + '            {"text": "Neutron activation analysis (NAA)'
)

# Insert new domains before closing ']'
old_end = "    ),\n]"
new_end = "    ),\n" + NEW_DOMAINS + "]"
content = content.replace(old_end, new_end, 1)

# ============================================================
# 2. Tighten hard difficulty
# ============================================================
content = content.replace(
    '"hard": {\n        "num_hypotheses": 5,\n        "num_evidence": 10,\n        "noise_level": 0.45,\n        "budget": 12,',
    '"hard": {\n        "num_hypotheses": 5,\n        "num_evidence": 10,\n        "noise_level": 0.50,\n        "budget": 10,'
)

with open('rctd_env/server/environment.py', 'w') as f:
    f.write(content)

# Count evidence templates
import re
count = len(re.findall(r'"text":', content.split('SCENARIO_TEMPLATES')[1].split('# Internal data')[0] if '# Internal data' in content else content.split('SCENARIO_TEMPLATES')[1]))
print(f"SUCCESS: File size: {len(content)} bytes")
print(f"Total evidence strings: {count}")
print(f"Domains: 7")
