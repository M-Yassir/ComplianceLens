import re
from pathlib import Path

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


def analyze_text_tokens(text: str, model_encoding: str = "cl100k_base") -> dict:
    """Calculates character count, word count, and token estimates for a given string."""
    char_count = len(text)
    word_count = len(text.split())

    # Heuristic estimates (4 chars/token or 1.3 tokens/word)
    approx_by_chars = round(char_count / 4)
    approx_by_words = round(word_count * 1.3)

    results = {
        "char_count": char_count,
        "word_count": word_count,
        "approx_by_chars": approx_by_chars,
        "approx_by_words": approx_by_words,
        "exact_tokens": None,
    }

    if HAS_TIKTOKEN:
        enc = tiktoken.get_encoding(model_encoding)
        results["exact_tokens"] = len(enc.encode(text))

    return results


def print_analysis(title: str, text: str):
    metrics = analyze_text_tokens(text)
    print(f"\n==========================================")
    print(f" Analysis: {title}")
    print(f"==========================================")
    print(f"Character Count : {metrics['char_count']:,}")
    print(f"Word Count      : {metrics['word_count']:,}")
    print(f"------------------------------------------")

    if metrics["exact_tokens"] is not None:
        print(f"Exact Tokens (cl100k_base): {metrics['exact_tokens']:,}")
    else:
        print("`tiktoken` package not installed.")
        print(f"Approx Tokens (Chars / 4)  : ~{metrics['approx_by_chars']:,}")
        print(f"Approx Tokens (Words * 1.3): ~{metrics['approx_by_words']:,}")


# =====================================================================
# 1. INPUT VARIABLE (Paste any text string here to analyze manually)
# =====================================================================
SAMPLE_TEXT = """
§ 7001. Definitions.
In addition to the definitions set forth in Civil Code section 1798.140, for purposes of
these regulations:
(a) “Agency” means the California Privacy Protection Agency established by Civil
Code section 1798.199.10 et seq.
(b) “Alternative Opt-out Link” means the alternative opt-out link that a business
may provide instead of posting the two separate “Do Not Sell or Share My
Personal Information” and “Limit the Use of My Sensitive Personal Information”
links as set forth in Civil Code section 1798.135, subdivision (a)(3), and
specified in section 7015.
(c)
“Attorney General” means the California Attorney General or any officer or
employee of the California Department of Justice acting under the authority of
the California Attorney General.
(d) “Authorized agent” means a natural person or a business entity that a
consumer has authorized to act on their behalf subject to the requirements set
forth in section 7063.
(e)
“Automated decisionmaking technology” or “ADMT” means any technology
that processes personal information and uses computation to replace human
decisionmaking or substantially replace human decisionmaking.
(1)
For purposes of this definition, to “substantially replace human
decisionmaking” means a business uses the technology’s output to make
a decision without human involvement. Human involvement requires the
human reviewer to:
(A) Know how to interpret and use the technology’s output to make the
decision;
(B) Review and analyze the output of the technology, and any other
information that is relevant to make or change the decision; and
(C) Have the authority to make or change the decision based on their
analysis in subsection (B).
(2) ADMT includes profiling that replaces human decisionmaking or
substantially replaces human decisionmaking.
(3) ADMT does not include web hosting, domain registration, networking,
caching, website-loading, data storage, firewalls, anti-virus, anti-malware,
spam- and robocall-filtering, spellchecking, calculators, databases, and
spreadsheets, provided that they do not replace human decisionmaking.
(f)
“Categories of sources” means types or groupings of persons or entities
from which a business collects personal information about consumers,
described with enough particularity to provide consumers with a meaningful
understanding of the type of person or entity. They may include the consumer
directly, advertising networks, internet service providers, data analytics
providers, government entities, operating systems and platforms, social
networks, and data brokers.
(g) “Categories of third parties” means types or groupings of third parties with
whom the business shares personal information, described with enough
particularity to provide consumers with a meaningful understanding of the
type of third party. They may include advertising networks, internet service
providers, data analytics providers, government entities, operating systems
and platforms, social networks, and data brokers.
(h)
“CCPA” means the California Consumer Privacy Act of 2018, Civil Code
section 1798.100 et seq.
(i)
“COPPA” means the Children’s Online Privacy Protection Act, 15 U.S.C.
sections 6501 to 6506 and 16 Code of Federal Regulations part 312.
(j)
“Cybersecurity audit” means the annual cybersecurity audit that every
business whose processing of consumers’ personal information presents
significant risk to consumers’ security as set forth in section 7120, subsection
(b), is required to complete.
(k)
“Cybersecurity program” means the policies, procedures, and practices that
protect personal information from unauthorized access, destruction, use,
modification, or disclosure; and protect against unauthorized activity resulting
in the loss of availability of personal information.
(l)
“Cybersecurity audit report” means the document that every business must
create as part of its cybersecurity audit. The cybersecurity audit report
includes the information set forth in section 7123, subsection (e).
(m) “Disproportionate effort” within the context of a business, service provider,
contractor, or third party responding to a consumer request means the time
and/or resources expended by the business, service provider, contractor, or
third party to respond to the individualized request significantly outweighs the
reasonably foreseeable impact to the consumer by not responding, taking into
account applicable circumstances, such as the size of the business, service
provider, contractor, or third party, the nature of the request, and the technical
limitations impacting their ability to respond. For example, responding to a
consumer request to know may require disproportionate effort when the
personal information that is the subject of the request is not in a searchable or
readily-accessible format, is maintained only for legal or compliance purposes,
is not sold or used for any commercial purpose, and there is no reasonably
foreseeable material impact to the consumer by not responding. By contrast,
the impact to the consumer of denying a request to correct inaccurate
information that the business uses and/or sells may outweigh the burden on
the business, service provider, contractor, or third party in honoring the
request when the reasonably foreseeable consequence of denying the request
would be the denial of services or opportunities to the consumer. A business,
service provider, contractor, or third party that has failed to put in place
adequate processes and procedures to receive and process consumer
requests in accordance with the CCPA and these regulations cannot claim that
responding to a consumer’s request requires disproportionate effort.
(n)
“Employment benefits” means retirement, health, and other benefit
programs, services, or products to which consumers and their dependents or
their beneficiaries receive access through the consumer’s employer.
(o)
“Employment-related information” means personal information that is
collected by the business about a natural person for the reasons identified in
Civil Code section 1798.145, subdivision (m)(1). The collection of employment-
related information, including for the purpose of administering employment
benefits, shall be considered a business purpose.
(p)
“Financial incentive” means a program, benefit, or other offering, including
payments to consumers, for the collection, retention, sale, or sharing of
personal information. Price or service differences are types of financial
incentives.
(q)
“First party” means a consumer-facing business with which the consumer
intends and expects to interact.
(r)
“Frictionless manner” means a business’s processing of an opt-out
preference signal that complies with the requirements set forth in section
7025, subsection (f).
(s)
“Information practices” means practices regarding the collection, use,
disclosure, sale, sharing, and retention of personal information.
(t)
“Information system” means the resources (e.g., network, hardware, and
software) organized for the processing of personal information or that can
provide access to personal information. The business’s information system
includes the resources organized for the business’s processing of personal
information, regardless of whether the business owns those resources.
(u)
“Multi-factor authentication” means authentication through verification of at
least two of the following types of authentication factors: (1) knowledge
factors, such as a password; (2) possession factors, such as a token; or (3)
inherence factors, such as a biometric characteristic.
(v)
“Nonbusiness” means a person or entity that does not meet the definition of
a “business” as defined in Civil Code section 1798.140, subdivision (d). For
example, government entities and many non-profits are
nonbusinesses because
they are not “organized or operated for the profit or financial
benefit of its shareholders or other owners.”
(w) “Notice at Collection” means the notice given by a business to a consumer
at or before the point at which a business collects personal information from
the consumer as required by Civil Code section 1798.100, subdivisions (a) and
(b), and specified in these regulations.
(x)
“Notice of Right to Limit” means the notice given by a business informing
consumers of their right to limit the use or disclosure of the consumer’s
sensitive personal information as required by Civil Code sections 1798.121 and
1798.135 and specified in these regulations.
(y)
“Notice of Right to Opt-out of Sale/Sharing” means the notice given by a
business informing consumers of their right to opt-out of the sale or sharing of
their personal information as required by Civil Code sections 1798.120 and
1798.135 and specified in these regulations.
(z)
“Notice of Financial Incentive” means the notice given by a business
explaining each financial incentive or price or service difference as required by
Civil Code section 1798.125, subdivision (b), and specified in these regulations.
(aa) “Opt-out preference signal” means a signal that is sent by a platform,
technology, or mechanism, on behalf of the consumer, that communicates the
consumer choice to opt-out of the sale and sharing of personal information and
that complies with the requirements set forth in section 7025, subsection (b).
(bb) “Penetration testing” means testing the security of an information system by
attempting to circumvent or defeat its security features by authorizing
attempted penetration of the information system.
(cc) “Performance at work” means the performance of job duties for which the
consumer has been hired or has applied to be hired. The following are not
“performance at work”: a consumer’s union membership or interest in
unionizing; a consumer’s interest in seeking other employment opportunities; a
consumer’s location when off-duty or on breaks; or a consumer’s use of a
personal account (e.g., email, text messages, or social media) unless solely to
prevent or limit the use of these accounts on the business’s information
system or to prevent the disclosure of confidential information.
(dd) “Performance in an educational program” means the performance of
coursework in an educational program in which the consumer is enrolled or has
applied to be enrolled. The following are not “performance in an educational
program”: a consumer’s use of a personal account (e.g., email, text messages,
or social media) unless solely to prevent or limit the use of these accounts on
the educational program provider’s information system, including to prevent
the disclosure of confidential information or to prevent cheating; or a
consumer’s location when they are not performing coursework.
(ee) “Physical or biological identification or profiling” means identifying or profiling
a consumer using automated measurements or analysis of their physical or
biological characteristics, or automated measurements or analysis of or
relating to their body. This includes using biometric information, vocal
intonation, facial expression, and gesture (e.g., to identify or infer emotion).
This does not include processing physical or biological characteristics that do
not identify, and cannot reasonably be linked with, a particular consumer.
(ff) “Price or service difference” means (1) any difference in the price or rate
charged for any goods or services to any consumer related to the collection,
retention, sale, or sharing of personal information, or (2) any difference in the
level or quality of any goods or services offered to any consumer related to the
collection, retention, sale, or sharing of personal information, including the
denial of goods or services to the consumer.
(gg) “Privacy policy,” as referred to in Civil Code sections 1798.130, subdivision
(a)(5), and 1798.135, subdivision (c)(2), means the statement that a business
shall make available to consumers describing the business’s online and offline
information practices, and the rights of consumers regarding their own
personal information.
(hh) “Privileged account” means any authorized user account (i.e., an account
designed to be used by an individual) or service account (i.e., an account
designed to be used only by a service, not by an individual) that can be used to
perform functions that other user accounts are not authorized to perform,
including but not limited to the ability to add, change, or remove other
accounts, or make configuration changes to an information system.
(ii)
“Profiling” means any form of automated processing of personal information to
evaluate certain personal aspects (including intelligence, ability, aptitude,
predispositions) relating to a natural person and in particular to analyze or
predict aspects concerning that natural person’s performance at work,
economic situation, health (including mental health), personal preferences,
interests, reliability, predispositions, behavior, location, or movements.
(jj)
“Request to access ADMT” means a consumer request that a business provide
information to the consumer about the business’s use of ADMT with respect to
the consumer, pursuant to Civil Code section 1798.185(a)(15) and Article 11.
(kk) “Request to appeal ADMT” means a consumer request to appeal the business’s
use of ADMT for a significant decision as set forth in section 7221, subsection
(b)(1).
(ll)
“Request to correct” means a consumer request that a business correct
inaccurate personal information that it maintains about the consumer,
pursuant to Civil Code section 1798.106.
(mm) “Request to delete” means a consumer request that a business delete
personal information about the consumer that the business has collected from
the consumer, pursuant to Civil Code section 1798.105.
(nn) “Request to know” means a consumer request that a business disclose
personal information that it has collected about the consumer pursuant to Civil
Code sections 1798.110 or 1798.115. It includes a request for any or all of the
following:
(1) Specific pieces of personal information that a business has collected
about the consumer;
(2) Categories of personal information it has collected about the consumer;
(3) Categories of sources from which the personal information is collected;
(4) Categories of personal information that the business sold, shared, or
disclosed for a business purpose about the consumer;
(5) Categories of third parties to whom the personal information was sold,
shared, or disclosed and
(6) The business or commercial purpose for collecting, selling, or sharing
personal information.
(oo) “Request to limit” means a consumer request that a business limit the use
and disclosure of the consumer’s sensitive personal information, pursuant to
Civil Code section 1798.121, subdivision (a).
(pp) “Request to opt-in to sale/sharing” means an action demonstrating that
the consumer has consented to the business’s sale or sharing of personal
information about the consumer by a parent or guardian of a consumer less
than 13 years of age or by a consumer at least 13 years of age.
(qq) “Request to opt-out of ADMT” means a consumer request that a business not
use ADMT with respect to the consumer, pursuant to Civil Code section
1798.185(a)(15) and Article 11.
(rr) “Request to opt-out of sale/sharing” means a consumer request that a
business neither sell nor share the consumer’s personal information to third
parties, pursuant to Civil Code section 1798.120, subdivision (a).
(ss) “Right to access ADMT” means a consumer’s right to request that a business
provide information to the consumer about the business’s use of ADMT with
respect to the consumer as set forth in Civil Code section 1798.185(a)(15) and
Article 11.
(tt) “Right to correct” means the consumer’s right to request that a business
correct inaccurate personal information that it maintains about the consumer
as set forth in Civil Code section 1798.106.
(uu) “Right to delete” means the consumer’s right to request that a business
delete any personal information about the consumer that the business has
collected from the consumer as set forth in Civil Code section 1798.105.
(vv) “Right to know” means the consumer’s right to request that a business
disclose personal information that it has collected, sold, or shared about the
consumer as set forth in Civil Code sections 1798.110 and 1798.115.
(ww) “Right to limit” means the consumer’s right to request that a business
limit the use and disclosure of a consumer’s sensitive personal information as
set forth in Civil Code section 1798.121.
(xx) “Right to opt-out of ADMT” means a consumer’s right to direct that a business
not use ADMT with respect to the consumer as set forth in Civil Code section
1798.185(a)(15) and Article 11.
(yy) “Right to opt-out of sale/sharing” means the consumer’s right to direct a
business that sells or shares personal information about the consumer to third
parties to stop doing so as set forth in Civil Code section 1798.120.
(zz) “Risk assessment report” means the document that every business that is
required to conduct a risk assessment must create as part of its risk
assessment. The risk assessment report includes the information set forth in
section 7152, subsections (a)(1)–(3), (6)–(9).
(aaa) “Sensitive location” means any of the following physical places: healthcare
facilities including hospitals, doctors’ offices, urgent care facilities, and
community health clinics; pharmacies; domestic violence shelters; food
pantries; housing/emergency shelters; educational institutions; political party
offices; legal services offices; union offices; and places of worship.
(bbb) “Sensitive personal information” means:
(1)
Personal information that reveals:
(A) A consumer’s social security, driver’s license, state identification
card, or passport number.
(B) A consumer’s account log-in, financial account, debit card, or credit
card number in combination with any required security or access
code, password, or credentials allowing access to an account.
(C) A consumer’s precise geolocation.
(D) A consumer’s racial or ethnic origin, citizenship or immigration status,
religious or philosophical beliefs, or union membership.
(E) The contents of a consumer’s mail, email, and text messages unless
the business is the intended recipient of the communication.
(F) A consumer’s genetic data.
(G) A consumer’s neural data, which means information that is generated
by measuring the activity of a consumer’s central or peripheral
nervous system, and that is not inferred from nonneural information.
(2)
The processing of biometric information for the purpose of uniquely
identifying a consumer.
(3) Personal information collected and analyzed concerning a consumer’s
health, sex life, or sexual orientation.
(4) Personal information of consumers that the business has actual
knowledge are less than 16 years of age. A business that willfully
disregards the consumer’s age shall be deemed to have had actual
knowledge of the consumer’s age.
Sensitive personal information does not include information that is “publicly
available” pursuant to Civil Code section 1798.140, subdivision (v)(2).
(ccc) “Signed” means that the written attestation, declaration, or permission
has either been physically signed or provided electronically in accordance with
the Uniform Electronic Transactions Act, Civil Code section 1633.1 et seq.
(ddd) “Significant decision” means a decision that results in the provision or denial
of financial or lending services, housing, education enrollment or opportunities,
employment or independent contracting opportunities or compensation, or
healthcare services. For purposes of this definition:
(1)
“Financial or lending services” means the extension of credit or a loan,
transmitting or exchanging funds, the provision of deposit or checking
accounts, check cashing, or installment payment plans.
(2) “Housing” means any building, structure, or portion thereof that is used or
occupied as, or designed, arranged, or intended to be used or occupied as,
a home, residence, or sleeping place by one or more consumers including
for permanent or temporary occupancy. The use of ADMT that provides or
denies housing to a consumer based solely on the availability or vacancy
of the housing or the successful receipt of payment for housing from the
consumer is not making a significant decision.
(3) “Education enrollment or opportunities” means:
(A) Admission or acceptance into academic or vocational programs;
(B) Educational credentials (e.g., a degree, diploma, or certificate); and
(C) Suspension and expulsion.
(4) “Employment or independent contracting opportunities or compensation”
means:
(A) Hiring;
(B) Allocation or assignment of work for employees; or salary, hourly or
per-assignment compensation, incentive compensation such as a
bonus, or another benefit (“allocation/assignment of work and
compensation”);
(C) Promotion; and
(D) Demotion, suspension, and termination.
(5) “Healthcare services” means services related to the diagnosis, prevention,
or treatment of human disease or impairment, or the assessment or care
of an individual's health.
(6) Significant decision does not include advertising to a consumer.
(eee) “Systematic observation” means methodical and regular or continuous
observation. This includes, for example, methodical and regular or continuous
observation using Wi-Fi or Bluetooth tracking, radio frequency identification,
drones, video or audio recording or live-streaming, technologies that enable
physical or biological identification or profiling; and geofencing, location
trackers, or license-plate recognition.
(fff) “Train” for the purposes of sections 7150, subsection (b)(6), and 7153 means
the process through which a technology discovers underlying patterns, learns
a series of actions, or is taught to generate a desired output. Examples of
training include adjusting the parameters of an algorithm used for ADMT,
improving the algorithm that determines how a machine-learning model learns,
and iterating the datasets fed into ADMT.
(ggg) “Third-party identity verification service” means a security process offered
by an independent third party that verifies the identity of the consumer making
a request to the business. Third-party identity verification services are subject
to the requirements set forth in Article 5 regarding requests to delete,
requests to correct, or requests to know.
(hhh) “Unstructured” as it relates to personal information means personal
information that is not organized in a pre-defined manner and could not be
retrieved or organized in a pre-defined manner without disproportionate effort
on behalf of the business, service provider, contractor, or third party.
(iii) “Value of the consumer’s data” means the value provided to the business
by the consumer’s data as calculated under section 7081.
(jjj) “Verify” means to determine that the consumer making a request to
delete, request to correct, request to know, or request to access ADMT is
the consumer about whom the business has collected information, or if that
consumer is less than 13 years of age, the consumer’s parent or legal guardian.
Note: Authority cited: Sections 1798.185, Civil Code. Reference: Sections
1798.100, 1798.105, 1798.106, 1798.110, 1798.115, 1798.120, 1798.121, 1798.125,
1798.130, 1798.135, 1798.140, 1798.145, 1798.150, 1798.155, 1798.175, 1798.185,
1798.199.40, 1798.199.45, 1798.199.50, and 1798.199.65, Civil Code.
"""

# =====================================================================
# 2. FILE BREAKDOWN UTILITY (Inspects token count per section)
# =====================================================================
def analyze_processed_file(file_path: Path):
    if not file_path.exists():
        print(f"\nFile not found: {file_path}")
        return

    content = file_path.read_text(encoding="utf-8")
    
    # Split content by Section (§) headers
    sections = re.split(r"(?=\n§\s*\d+)", content)

    print(f"\n==========================================")
    print(f" SECTION BREAKDOWN FOR: {file_path.name}")
    print(f"==========================================")
    print(f"{'Section Title / ID':<30} | {'Words':<8} | {'Tokens':<8}")
    print("-" * 52)

    total_doc_tokens = 0

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        # Extract title line
        first_line = sec.split("\n")[0][:28]
        metrics = analyze_text_tokens(sec)
        token_count = metrics["exact_tokens"] or metrics["approx_by_words"]
        total_doc_tokens += token_count

        print(f"{first_line:<30} | {metrics['word_count']:<8,} | {token_count:<8,}")

    print("-" * 52)
    print(f"{'TOTAL DOCUMENT':<30} | {'-':<8} | {total_doc_tokens:<8,}\n")


if __name__ == "__main__":
    # 1. Run analysis on the variable above
    print_analysis("Manual String Variable", SAMPLE_TEXT)

    # 2. Run analysis on your actual processed text file
    PROJECT_ROOT = Path(__file__).resolve().parent
    PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "version_N.txt"
    
    analyze_processed_file(PROCESSED_FILE)