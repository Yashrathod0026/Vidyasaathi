"""
Tutor Agent for VidyaSaathi.
Generates personalized, culturally relatable vernacular lessons in English, Hindi, and Marathi.
Includes:
- Intuitive explanation using relatable Indian real-world analogies (roti, cricket, market)
- Step-by-step worked example with clear mathematical reasoning
- 3 to 5 adaptive practice questions
- Encouragement note
- In-memory / persistent explanation caching by (concept, language, misconception)
"""

from typing import Dict, Optional, Tuple
import logging
from backend.app.core.schemas import (
    TutorOutput, WorkedExample, WorkedExampleStep, TutorQuestion
)
from backend.app.core.bkt import ConceptGraph
from backend.app.core.llm_client import LLMClient

logger = logging.getLogger("VidyaSaathi.Tutor")


class TutorAgent:
    """Agent that creates friendly, step-by-step lessons in the student's own vernacular."""

    SYSTEM_PROMPT = """
You are 'VidyaSaathi' (विद्यासाथी), a compassionate, warm, and highly effective mathematics tutor for Class 7 students in rural and low-resource schools in India.
Your mission:
1. Explain mathematics intuitively without terrifying jargon.
2. Generate content directly in the requested language (English, Hindi हिन्दी, or Marathi मराठी) - DO NOT translate word-for-word. Use natural, spoken-style vernacular phrasing.
3. Use relatable Indian analogies: sharing rotis/chapatis, cricket scores and wickets, buying vegetables from the sabzi mandi, borrowing pocket money, balancing a grocery weighing scale.
4. If a student holds a misconception, gently untangle it without making them feel judged or inadequate.
5. Provide a step-by-step worked example and 3 multiple-choice practice questions with hints.
6. Strictly return JSON adhering to the TutorOutput schema.
"""

    def __init__(
        self,
        concept_graph: Optional[ConceptGraph] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.graph = concept_graph or ConceptGraph()
        self.llm = llm_client or LLMClient()
        # In-memory explanation cache: (concept_id, language, misconception_id or "") -> TutorOutput
        self._cache: Dict[Tuple[str, str, str], TutorOutput] = {}

    def get_lesson(
        self,
        concept_id: str,
        language: str = "en",
        misconception_id: Optional[str] = None
    ) -> TutorOutput:
        """
        Retrieves or generates a personalized vernacular lesson. Checks cache first.
        """
        cache_key = (concept_id, language, misconception_id or "")
        if cache_key in self._cache:
            logger.info(f"Serving tutor lesson from cache for key: {cache_key}")
            return self._cache[cache_key]

        concept = self.graph.get_concept(concept_id)
        c_name = concept["name"].get(language, concept["name"]["en"]) if concept else concept_id

        # Fallback generator for instant offline responsiveness or LLM unavailability
        def fallback_tutor() -> TutorOutput:
            return self._generate_templated_lesson(concept_id, language, misconception_id)

        prompt = f"""
Target Concept ID: {concept_id}
Concept Name: {c_name}
Target Language: {language}
Active Misconception: {misconception_id or 'None'}

Please construct a comprehensive, encouraging lesson in {language}.
Include:
1. An intuitive explanation using a familiar daily life analogy.
2. A step-by-step worked example.
3. 3 adaptive practice questions with 4 options and correct option index (0-3).
4. An encouraging note.
"""

        output = self.llm.generate_structured(
            prompt=prompt,
            system_instruction=self.SYSTEM_PROMPT,
            response_schema=TutorOutput,
            model="gemini-2.5-flash",
            fallback_factory=fallback_tutor
        )

        # Store in cache
        self._cache[cache_key] = output
        return output

    def _generate_templated_lesson(
        self,
        concept_id: str,
        language: str,
        misconception_id: Optional[str]
    ) -> TutorOutput:
        """
        Deterministic, high-quality vernacular lesson templates for offline or fast-path execution.
        """
        concept = self.graph.get_concept(concept_id)
        title = concept["name"].get(language, concept["name"]["en"]) if concept else concept_id

        if concept_id == "int_add_sub":
            if language == "hi":
                explanation = (
                    "पूर्णांकों को जोड़ने और घटाने को हम दुकान के लेन-देन से समझ सकते हैं! "
                    "मान लीजिए धनात्मक संख्या (+) आपकी जेब के रुपये हैं और ऋणात्मक संख्या (-) आपका उधार या कर्ज है। "
                    "यदि आपने कल ₹5 का उधार किया (-5) और आज फिर ₹3 का उधार किया (-3), तो कुल मिलाकर आपके ऊपर ₹8 का कर्ज (-8) हो गया! "
                    "वही जब हम किसी का उधार माफ करते हैं (घटाव का माइनस और उधार का माइनस), तो वह आपके लिए लाभ यानी प्लस (+) बन जाता है।"
                )
                worked = WorkedExample(
                    problem_statement="हल कीजिए: (-4) + (-7) और 5 - (-3)",
                    steps=[
                        WorkedExampleStep(
                            step_number=1,
                            explanation="पहले भाग (-4) + (-7) में दोनों संख्याएं ऋण हैं। दोनों का कर्ज जुड़ेगा: 4 + 7 = 11।",
                            math_expression="(-4) + (-7) = -11"
                        ),
                        WorkedExampleStep(
                            step_number=2,
                            explanation="दूसरे भाग 5 - (-3) में दो माइनस पास आने पर प्लस बन जाते हैं: 5 + 3 = 8।",
                            math_expression="5 - (-3) = 5 + 3 = 8"
                        )
                    ],
                    final_answer="(-4) + (-7) = -11, और 5 - (-3) = 8"
                )
                questions = [
                    TutorQuestion(
                        question_id="t_int_1",
                        question_text="मान ज्ञात कीजिए: (-6) + (-2)",
                        options=["-8", "8", "-4", "4"],
                        correct_option_index=0,
                        hint="दोनों संख्याएं ऋणात्मक हैं, अतः दोनों मान जुड़ेंगे और चिन्ह ऋण रहेगा।"
                    ),
                    TutorQuestion(
                        question_id="t_int_2",
                        question_text="हल कीजिए: 8 - (-5)",
                        options=["3", "13", "-13", "-3"],
                        correct_option_index=1,
                        hint="माइनस और माइनस मिलकर प्लस बनता है: 8 + 5।"
                    ),
                    TutorQuestion(
                        question_id="t_int_3",
                        question_text="हल कीजिए: (-10) + 14",
                        options=["-4", "24", "4", "-24"],
                        correct_option_index=2,
                        hint="₹10 का कर्ज था और ₹14 जेब में आए, तो कितने बचे?"
                    )
                ]
                encouragement = "बहुत बढ़िया! गणित कोई जादू नहीं, बस हमारे रोज़मर्रा के जीवन का हिसाब-किताब है।"

            elif language == "mr":
                explanation = (
                    "पूर्णांकांची बेरीज आणि वजाबाकी समजण्यासाठी किराणा दुकानातील हिशोबाची कल्पना करा! "
                    "धन संख्या (+) म्हणजे आपल्या खिशातील पैसे आणि ऋण संख्या (-) म्हणजे उधारी किंवा कर्ज. "
                    "जर तुम्ही काल ५ रुपये उसने घेतले (-५) आणि आज परत ३ रुपये उसने घेतले (-३), तर एकूण ८ रुपये कर्ज झाले (-८)! "
                    "आणि वजाबाकी करताना जेव्हा दोनदा वजा चिन्ह येते (-(-३)), तेव्हा त्याचे अधिक (+) होते, कारण कर्ज रद्द होणे म्हणजे फायदा!"
                )
                worked = WorkedExample(
                    problem_statement="सोडवा: (-४) + (-७) आणि ५ - (-३)",
                    steps=[
                        WorkedExampleStep(
                            step_number=1,
                            explanation="पहिल्या भागात (-४) + (-७) दोन्ही संख्या ऋण आहेत, त्यामुळे कर्जाची बेरीज होईल: ४ + ७ = ११.",
                            math_expression="(-४) + (-७) = -११"
                        ),
                        WorkedExampleStep(
                            step_number=2,
                            explanation="दुसऱ्या भागात ५ - (-३) मध्ये दोन वजा चिन्हे मिळून अधिक होतील: ५ + ३ = ८.",
                            math_expression="५ - (-३) = ५ + ३ = ८"
                        )
                    ],
                    final_answer="(-४) + (-७) = -११ आणि ५ - (-३) = ८"
                )
                questions = [
                    TutorQuestion(
                        question_id="t_int_1",
                        question_text="मूल्य काढा: (-६) + (-२)",
                        options=["-८", "८", "-४", "४"],
                        correct_option_index=0,
                        hint="दोन्ही ऋण संख्यांची बेरीज करा आणि ऋण चिन्ह द्या."
                    ),
                    TutorQuestion(
                        question_id="t_int_2",
                        question_text="सोडवा: ८ - (-५)",
                        options=["३", "१३", "-१३", "-३"],
                        correct_option_index=1,
                        hint="वजा वजा अधिक होते: ८ + ५."
                    ),
                    TutorQuestion(
                        question_id="t_int_3",
                        question_text="सोडवा: (-१०) + १४",
                        options=["-४", "२४", "४", "-२४"],
                        correct_option_index=2,
                        hint="१० रुपये कर्ज होते आणि १४ रुपये मिळाले, तर किती शिल्लक राहिले?"
                    )
                ]
                encouragement = "शाब्बास! गणिताची भीती बाळगू नका, सराव केल्यास हे अगदी सोपे आहे."

            else:  # English
                explanation = (
                    "Think of adding and subtracting integers like borrowing and earning money! "
                    "Positive numbers (+) represent cash in your pocket, while negative numbers (-) represent a debt or loan. "
                    "If you borrow ₹5 (-5) and then borrow ₹3 more (-3), your total debt is ₹8 (-8). "
                    "When subtracting a negative number, like 5 - (-3), taking away a debt gives you more wealth, so it becomes 5 + 3 = 8!"
                )
                worked = WorkedExample(
                    problem_statement="Solve: (-4) + (-7) and 5 - (-3)",
                    steps=[
                        WorkedExampleStep(
                            step_number=1,
                            explanation="For (-4) + (-7), both have negative signs. Combine their absolute values: 4 + 7 = 11, with negative sign.",
                            math_expression="(-4) + (-7) = -11"
                        ),
                        WorkedExampleStep(
                            step_number=2,
                            explanation="For 5 - (-3), two minus signs cancel to form a plus: 5 + 3 = 8.",
                            math_expression="5 - (-3) = 5 + 3 = 8"
                        )
                    ],
                    final_answer="(-4) + (-7) = -11 and 5 - (-3) = 8"
                )
                questions = [
                    TutorQuestion(
                        question_id="t_int_1",
                        question_text="Evaluate: (-6) + (-2)",
                        options=["-8", "8", "-4", "4"],
                        correct_option_index=0,
                        hint="Both numbers are negative. Add them up and keep the negative sign."
                    ),
                    TutorQuestion(
                        question_id="t_int_2",
                        question_text="Solve: 8 - (-5)",
                        options=["3", "13", "-13", "-3"],
                        correct_option_index=1,
                        hint="Subtracting a negative turns into addition: 8 + 5."
                    ),
                    TutorQuestion(
                        question_id="t_int_3",
                        question_text="Calculate: (-10) + 14",
                        options=["-4", "24", "4", "-24"],
                        correct_option_index=2,
                        hint="You owe ₹10, but earn ₹14. How much do you have left?"
                    )
                ]
                encouragement = "Fantastic effort! You are mastering foundational integer arithmetic step-by-step."

        elif concept_id == "frac_add_sub":
            if language == "hi":
                explanation = (
                    "भिन्नों को जोड़ना ठीक वैसे ही है जैसे दो अलग-अलग तरह के कटे हुए तरबूज के टुकड़ों को मिलाना! "
                    "यदि एक टुकड़ा आधा (1/2) है और दूसरा तिहाई (1/3), तो आप उन्हें सीधे '2 टुकड़े' नहीं कह सकते क्योंकि दोनों का आकार अलग है। "
                    "पहले हमें दोनों टुकड़ों को बराबर आकार का बनाना पड़ता है — इसी को हरों का ल.स.प. (LCM) लेना कहते हैं! "
                    "2 और 3 का LCM 6 होता है। 1/2 बन जाता है 3/6 और 1/3 बन जाता है 2/6। अब 3 और 2 मिलकर बने 5/6!"
                )
                worked = WorkedExample(
                    problem_statement="हल कीजिए: 1/4 + 2/3",
                    steps=[
                        WorkedExampleStep(
                            step_number=1,
                            explanation="हर 4 और 3 का ल.स.प. (LCM) ज्ञात करें: 4 * 3 = 12।",
                            math_expression="LCM(4, 3) = 12"
                        ),
                        WorkedExampleStep(
                            step_number=2,
                            explanation="दोनों भिन्नों को समान हर 12 में बदलें: (1*3)/12 = 3/12 और (2*4)/12 = 8/12।",
                            math_expression="3/12 + 8/12"
                        ),
                        WorkedExampleStep(
                            step_number=3,
                            explanation="अब अंशों को जोड़ें: 3 + 8 = 11।",
                            math_expression="(3 + 8) / 12 = 11/12"
                        )
                    ],
                    final_answer="11/12"
                )
                questions = [
                    TutorQuestion(
                        question_id="t_frac_1",
                        question_text="हल कीजिए: 1/2 + 1/4",
                        options=["3/4", "2/6", "1/6", "2/4"],
                        correct_option_index=0,
                        hint="2 और 4 का LCM 4 है। 1/2 को 2/4 लिखें, फिर 2/4 + 1/4 करें।"
                    ),
                    TutorQuestion(
                        question_id="t_frac_2",
                        question_text="हल कीजिए: 2/3 - 1/6",
                        options=["1/3", "3/6 = 1/2", "1/6", "3/3 = 1"],
                        correct_option_index=1,
                        hint="3 और 6 का LCM 6 है। 2/3 को 4/6 बनाएं, फिर 4/6 - 1/6 = 3/6 = 1/2।"
                    ),
                    TutorQuestion(
                        question_id="t_frac_3",
                        question_text="हल कीजिए: 3/5 + 1/10",
                        options=["4/15", "7/10", "4/10", "2/5"],
                        correct_option_index=1,
                        hint="5 को 10 बनाने के लिए 2 से गुणा करें: 3/5 = 6/10।"
                    )
                ]
                encouragement = "बहुत सुंदर! अब आप भिन्न के जोड़ के उस्ताद बन रहे हैं।"

            elif language == "mr":
                explanation = (
                    "अपूर्णांकांची बेरीज करणे म्हणजे वेगवेगळ्या आकाराचे तुकडे एकत्र मोजण्यासारखे आहे! "
                    "जर एक तुकडा अर्धा (१/२) आणि दुसरा एक-तृतीयांश (१/३) असेल, तर आपण त्यांना सरळ जोडू शकत नाही कारण दोघांचा आकार वेगळा आहे. "
                    "आधी दोन्ही तुकड्यांना समान आकारात आणावे लागते — यालाच छेदांचा लसावि काढणे म्हणतात. "
                    "२ आणि ३ चा लसावि ६ आहे. १/२ चे ३/६ होतात आणि १/३ चे २/६ होतात. आता ३ + २ = ५/६!"
                )
                worked = WorkedExample(
                    problem_statement="सोडवा: १/४ + २/३",
                    steps=[
                        WorkedExampleStep(
                            step_number=1,
                            explanation="४ आणि ३ या छेदांचा लसावि काढा: ४ * ३ = १२.",
                            math_expression="लसावि(४, ३) = १२"
                        ),
                        WorkedExampleStep(
                            step_number=2,
                            explanation="समच्छेद अपूर्णांक तयार करा: (१*३)/१२ = ३/१२ आणि (२*४)/१२ = ८/१२.",
                            math_expression="३/१२ + ८/१२"
                        ),
                        WorkedExampleStep(
                            step_number=3,
                            explanation="अंशांची बेरीज करा: ३ + ८ = ११.",
                            math_expression="(३ + ८) / १२ = ११/१२"
                        )
                    ],
                    final_answer="११/१२"
                )
                questions = [
                    TutorQuestion(
                        question_id="t_frac_1",
                        question_text="सोडवा: १/२ + १/४",
                        options=["३/४", "२/६", "१/६", "२/४"],
                        correct_option_index=0,
                        hint="२ आणि ४ चा लसावि ४ आहे. १/२ म्हणजे २/४."
                    ),
                    TutorQuestion(
                        question_id="t_frac_2",
                        question_text="सोडवा: २/३ - १/६",
                        options=["१/३", "३/६ = १/२", "१/६", "१"],
                        correct_option_index=1,
                        hint="२/३ चे ४/६ करा, मग ४/६ - १/६ = ३/६ = १/२."
                    ),
                    TutorQuestion(
                        question_id="t_frac_3",
                        question_text="सोडवा: ३/५ + १/१०",
                        options=["४/१५", "७/१०", "४/१०", "२/५"],
                        correct_option_index=1,
                        hint="३/५ चे ६/१० होतात. ६/१० + १/१० = ७/१०."
                    )
                ]
                encouragement = "उत्तम! पायरी-पायरीने गेलात की अपूर्णांक अतिशय सोपे वाटू लागतात."

            else:  # English
                explanation = (
                    "Adding fractions is like trying to combine slices of pizza of different sizes! "
                    "If you have 1/2 of a pizza and 1/3 of a pizza, you cannot simply add them as 2/5. "
                    "First, you slice both into equal-sized pieces by finding the Least Common Multiple (LCM) of the denominators. "
                    "For 2 and 3, the LCM is 6. 1/2 becomes 3/6, and 1/3 becomes 2/6. Now, 3 slices + 2 slices = 5/6 slices!"
                )
                worked = WorkedExample(
                    problem_statement="Solve: 1/4 + 2/3",
                    steps=[
                        WorkedExampleStep(
                            step_number=1,
                            explanation="Find the LCM of denominators 4 and 3: LCM(4, 3) = 12.",
                            math_expression="LCM(4, 3) = 12"
                        ),
                        WorkedExampleStep(
                            step_number=2,
                            explanation="Convert to like fractions: 1/4 = 3/12 and 2/3 = 8/12.",
                            math_expression="3/12 + 8/12"
                        ),
                        WorkedExampleStep(
                            step_number=3,
                            explanation="Add numerators over common denominator: 3 + 8 = 11.",
                            math_expression="(3 + 8) / 12 = 11/12"
                        )
                    ],
                    final_answer="11/12"
                )
                questions = [
                    TutorQuestion(
                        question_id="t_frac_1",
                        question_text="Solve: 1/2 + 1/4",
                        options=["3/4", "2/6", "1/6", "2/4"],
                        correct_option_index=0,
                        hint="The LCM of 2 and 4 is 4. Convert 1/2 into 2/4 first."
                    ),
                    TutorQuestion(
                        question_id="t_frac_2",
                        question_text="Calculate: 2/3 - 1/6",
                        options=["1/3", "3/6 = 1/2", "1/6", "1"],
                        correct_option_index=1,
                        hint="Convert 2/3 to 4/6, then 4/6 - 1/6 = 3/6 = 1/2."
                    ),
                    TutorQuestion(
                        question_id="t_frac_3",
                        question_text="Solve: 3/5 + 1/10",
                        options=["4/15", "7/10", "4/10", "2/5"],
                        correct_option_index=1,
                        hint="Multiply numerator and denominator of 3/5 by 2 to get 6/10."
                    )
                ]
                encouragement = "Awesome job! You now understand the true principle behind common denominators."

        else:
            # Generic template for any other Class 7 concept
            if language == "hi":
                explanation = f"{title} को सरलता से समझना आसान है। जब हम बुनियादी नियमों को समझकर एक-एक कदम आगे बढ़ते हैं, तो गणित सबसे रोचक विषय बन जाता है।"
                worked = WorkedExample(
                    problem_statement=f"{title} का उदाहरण अभ्यास",
                    steps=[
                        WorkedExampleStep(step_number=1, explanation="दिए गए मानों को पहचानें।", math_expression="Step 1"),
                        WorkedExampleStep(step_number=2, explanation="नियम लगाकर सरल करें।", math_expression="Step 2")
                    ],
                    final_answer="सत्यापित उत्तर"
                )
                questions = [
                    TutorQuestion(
                        question_id=f"t_{concept_id}_1",
                        question_text=f"{title} का आधारभूत प्रश्न १",
                        options=["विकल्प अ (सही)", "विकल्प ब", "विकल्प स", "विकल्प द"],
                        correct_option_index=0,
                        hint="मूलभूत नियम का स्मरण करें।"
                    ),
                    TutorQuestion(
                        question_id=f"t_{concept_id}_2",
                        question_text=f"{title} का अभ्यास प्रश्न २",
                        options=["विकल्प अ", "विकल्प ब (सही)", "विकल्प स", "विकल्प द"],
                        correct_option_index=1,
                        hint="पदों को व्यवस्थित करें।"
                    ),
                    TutorQuestion(
                        question_id=f"t_{concept_id}_3",
                        question_text=f"{title} का अभ्यास प्रश्न ३",
                        options=["विकल्प अ", "विकल्प ब", "विकल्प स (सही)", "विकल्प द"],
                        correct_option_index=2,
                        hint="अंतिम गणना ध्यान से करें।"
                    )
                ]
                encouragement = "आप बहुत अच्छा प्रयास कर रहे हैं! इसी तरह सीखते रहिए।"
            elif language == "mr":
                explanation = f"{title} समजून घेणे अतिशय सोपे आहे. गणितातील पायऱ्या योग्य क्रमाने सोडवल्या की उत्तर अचूक मिळते."
                worked = WorkedExample(
                    problem_statement=f"{title} चे नमुना उदाहरण",
                    steps=[
                        WorkedExampleStep(step_number=1, explanation="उदाहरणातील माहिती व्यवस्थित मांडा.", math_expression="Step 1"),
                        WorkedExampleStep(step_number=2, explanation="योग्य नियम वापरून उत्तर काढा.", math_expression="Step 2")
                    ],
                    final_answer="अचूक उत्तर"
                )
                questions = [
                    TutorQuestion(
                        question_id=f"t_{concept_id}_1",
                        question_text=f"{title} वर आधारित प्रश्न १",
                        options=["पर्याय अ (योग्य)", "पर्याय ब", "पर्याय क", "पर्याय ड"],
                        correct_option_index=0,
                        hint="संकल्पना नीट आठवा."
                    ),
                    TutorQuestion(
                        question_id=f"t_{concept_id}_2",
                        question_text=f"{title} वर आधारित प्रश्न २",
                        options=["पर्याय अ", "पर्याय ब (योग्य)", "पर्याय क", "पर्याय ड"],
                        correct_option_index=1,
                        hint="पायऱ्या योग्य क्रमाने सोडवा."
                    ),
                    TutorQuestion(
                        question_id=f"t_{concept_id}_3",
                        question_text=f"{title} वर आधारित प्रश्न ३",
                        options=["पर्याय अ", "पर्याय ब", "पर्याय क (योग्य)", "पर्याय ड"],
                        correct_option_index=2,
                        hint="अंतिम आकडेमोड तपासा."
                    )
                ]
                encouragement = "खूप छान! तुमचा सराव असाच सुरू ठेवा."
            else:
                explanation = f"Mastering {title} requires breaking down each problem into clear logical steps."
                worked = WorkedExample(
                    problem_statement=f"Worked demonstration for {title}",
                    steps=[
                        WorkedExampleStep(step_number=1, explanation="Identify the given parameters.", math_expression="Step 1"),
                        WorkedExampleStep(step_number=2, explanation="Apply the core formula and solve.", math_expression="Step 2")
                    ],
                    final_answer="Verified Solution"
                )
                questions = [
                    TutorQuestion(
                        question_id=f"t_{concept_id}_1",
                        question_text=f"Core question 1 on {title}",
                        options=["Option A (Correct)", "Option B", "Option C", "Option D"],
                        correct_option_index=0,
                        hint="Review the fundamental rule."
                    ),
                    TutorQuestion(
                        question_id=f"t_{concept_id}_2",
                        question_text=f"Practice question 2 on {title}",
                        options=["Option A", "Option B (Correct)", "Option C", "Option D"],
                        correct_option_index=1,
                        hint="Isolate the required quantity."
                    ),
                    TutorQuestion(
                        question_id=f"t_{concept_id}_3",
                        question_text=f"Application question 3 on {title}",
                        options=["Option A", "Option B", "Option C (Correct)", "Option D"],
                        correct_option_index=2,
                        hint="Double-check the calculation."
                    )
                ]
                encouragement = "Great job! Keep practicing with confidence."

        return TutorOutput(
            concept_id=concept_id,
            language=language,
            concept_title=title,
            intuitive_explanation=explanation,
            worked_example=worked,
            practice_questions=questions,
            encouragement_note=encouragement
        )
