# -*- coding: utf-8 -*-
"""
Legal-only evaluation (Egyptian PDPL 151/2020 + executive regulations + DPO /
breach / consent / RoPA ... docs). Reuses the eval50 harness but with 50 legal
questions and writes EVAL_REPORT_UPDATED.md.

Run inside the backend container:
    docker compose exec -T backend python eval_legal.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import eval50

# 45 in-corpus legal questions + 5 legal-but-out-of-corpus (refusal tests).
eval50.QUESTIONS = [
    # ---- Penalties ----
    ("ما هي عقوبة مخالفة قانون حماية البيانات الشخصية المصري؟", "Penalties", True),
    ("ما نص المادة 73 من قانون حماية البيانات الشخصية؟", "Penalties", True),
    ("ما هي عقوبة الامتناع عن تمكين صاحب البيانات من ممارسة حقوقه؟", "Penalties", True),
    ("ما هي عقوبة نقل البيانات الشخصية للخارج بالمخالفة للقانون؟", "Penalties", True),
    ("ما هي غرامة معالجة البيانات الحساسة دون تصريح؟", "Penalties", True),
    # ---- DPO ----
    ("ما هي مسؤوليات مسؤول حماية البيانات (DPO)؟", "DPO", True),
    ("متى يجب تعيين مسؤول لحماية البيانات؟", "DPO", True),
    ("What are the responsibilities of a Data Protection Officer?", "DPO", True),
    # ---- Data subject rights ----
    ("ما هي حقوق صاحب البيانات الشخصية؟", "Rights", True),
    ("كيف يمارس صاحب البيانات حقه في المحو؟", "Rights", True),
    ("What rights does a data subject have under the law?", "Rights", True),
    # ---- Lawful basis / consent ----
    ("ما هو الأساس القانوني لمعالجة البيانات الشخصية؟", "LawfulBasis", True),
    ("What is the lawful basis for processing personal data?", "LawfulBasis", True),
    ("ما هي شروط الموافقة الصحيحة على معالجة البيانات؟", "Consent", True),
    ("ما شروط الحصول على موافقة القاصر؟", "Consent", True),
    # ---- Breach notification ----
    ("ما هي المدة المحددة للإخطار بخرق البيانات الشخصية للمركز؟", "Breach", True),
    ("متى يجب إخطار صاحب البيانات بخرق بياناته؟", "Breach", True),
    ("ما البيانات التي يجب تضمينها عند الإخطار بالخرق؟", "Breach", True),
    ("What is a personal data breach and how must it be reported?", "Breach", True),
    # ---- RoPA ----
    ("ما هو سجل أنشطة المعالجة (RoPA) وما محتواه؟", "RoPA", True),
    ("What must a Record of Processing Activities contain?", "RoPA", True),
    # ---- Licenses & permits ----
    ("ما هي التراخيص والتصاريح المطلوبة لمعالجة البيانات؟", "Licenses", True),
    ("ما شروط منح ترخيص أو تصريح بمعالجة البيانات؟", "Licenses", True),
    ("What licenses or permits are required to process personal data?", "Licenses", True),
    # ---- Principles ----
    ("ما هي مبادئ حماية البيانات الشخصية؟", "Principles", True),
    ("What are the principles of personal data protection?", "Principles", True),
    # ---- PDPC ----
    ("ما هو دور مركز حماية البيانات الشخصية (PDPC)؟", "PDPC", True),
    ("ما هي اختصاصات مركز حماية البيانات الشخصية؟", "PDPC", True),
    ("ما إجراءات تقديم شكوى لمركز حماية البيانات؟", "PDPC", True),
    ("What is the role of the PDPC?", "PDPC", True),
    # ---- Controller / Processor ----
    ("ما هي التزامات المتحكم في البيانات الشخصية؟", "Controller", True),
    ("ما هي التزامات المعالج للبيانات الشخصية؟", "Processor", True),
    ("هل يجوز الاستعانة بطرف ثالث لمعالجة البيانات وما شروطه؟", "Processor", True),
    ("ما الفرق بين المتحكم والمعالج في الالتزامات؟", "Controller", True),
    # ---- Definitions / scope ----
    ("ما تعريف البيانات الشخصية في القانون؟", "Definitions", True),
    ("ما تعريف المعالجة في القانون؟", "Definitions", True),
    ("ما هي البيانات الشخصية الحساسة وكيف تُعالَج؟", "Sensitive", True),
    ("ما هي حالات الإعفاء من تطبيق القانون؟", "Scope", True),
    # ---- Cross-border / marketing / security / notice ----
    ("ما هي ضوابط نقل البيانات الشخصية خارج البلاد؟", "CrossBorder", True),
    ("ما هي شروط التسويق الإلكتروني المباشر؟", "Marketing", True),
    ("What consent is required for electronic direct marketing?", "Marketing", True),
    ("ما الذي يجب أن يتضمنه إشعار الخصوصية؟", "PrivacyNotice", True),
    ("ما هي الضمانات التقنية والتنظيمية المطلوبة لحماية البيانات؟", "Security", True),
    ("ما هي مدة الاحتفاظ بالبيانات الشخصية؟", "Retention", True),
    ("What are the data protection compliance requirements for a company?", "Compliance", True),
    # ---- Legal but OUT-OF-CORPUS (should refuse, not hallucinate) ----
    ("ما هي الغرامات القصوى في لائحة GDPR الأوروبية؟", "OOC", False),
    ("What are the penalties under the California CCPA?", "OOC", False),
    ("ما نص المادة 200 من قانون حماية البيانات المصري؟", "OOC", False),
    ("ما أحكام قانون العمل المصري بخصوص الإجازات السنوية؟", "OOC", False),
    ("What does the Saudi PDPL require about data localization?", "OOC", False),
]

eval50.OUTPUT_FILE = "EVAL_REPORT_UPDATED.md"
eval50.REPORT_TITLE = "RAG Evaluation Report (UPDATED — Legal Corpus) — 50 Legal Questions"

if __name__ == "__main__":
    eval50.main()
