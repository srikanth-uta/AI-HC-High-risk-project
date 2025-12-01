# TASK


Recent advances in large language models (LLMs) have enabled new approaches for personalized health assessment and decision support. Traditional clinical screening tools require structured input, predefined questionnaires, and manual interpretation by clinicians. In contrast, foundation models such as OpenBioLLM and m42-health/Llama3-Med42-8B demonstrate strong performance on biomedical reasoning tasks and provide a flexible interface for interacting with unstructured user input. These capabilities offer an opportunity to design intelligent systems that can collect, interpret, and respond to individual-level health information in real time.

In this work, we explore an LLM-driven framework for interactive personal health assessment. The proposed system allows an individual to provide health-related information through a dynamic, model-generated form. During initial interaction, the LLM elicits key health features including recent body mass index (BMI), blood glucose levels, weight, blood pressure, heart rate, cholesterol levels, and blood oxygen saturation. Users may also disclose relevant medical conditions or symptoms. Once submitted, these inputs are incorporated into the model's context or used to update a fine-tunable component of the system.

The LLM then generates personalized responses adapted to the user's profile. These responses may include: (i) contextualized explanations of potential health risks; (ii) evidence-aligned wellness suggestions; and (iii) lifestyle or preventive recommendations derived from medical knowledge encoded in the model. Additionally, when requested, the system can assist with geolocation-aware recommendations, such as identifying nearby healthcare facilities, laboratories, or emergency services.

This work aims to investigate the feasibility, benefits, and limitations of such an LLM-based health assessment system. We discuss the technical design, data considerations, and safety constraints necessary to ensure accuracy, reliability, and user privacy. Ultimately, this study provides insights into how domain-specialized LLMs may support personalized health guidance while maintaining alignment with clinical best practices and responsible AI principles.

# GOAL
Build a patient-facing health assistant LLM that
	1.	reads structured clinical metrics + text reports,
	2.	explains them in simple language, and
	3.	gives lifestyle + next-step guidance (no diagnosis/prescriptions).

Using Llama3-OpenBioLLM-8B or 70B with synthetic guideline-based training.

We train a biomedical LLM (Llama3-OpenBioLLM-8B) on a synthetic dataset containing structured clinical metrics and free-text patient context.
We perform Supervised Fine-Tuning (SFT) using QLoRA, evaluate performance using training/validation loss, manual clinical scoring, and qualitative examples, and document challenges and future work.
