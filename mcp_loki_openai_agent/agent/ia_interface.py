import openai

class OpenAIInterface:
    def __init__(self, model="gpt-5-mini"):
        self.model = model

    def suggest_refactoring(self, issue, code_snippet):
        prompt = f'''
Tu es un expert Fortran. Voici un extrait de code :
{code_snippet}

Issue détectée : {issue['debt_type']} aux lignes {issue['line_range']}.
Propose un refactoring clair et concis, si possible avec adaptation GPU (OpenACC).
Répond uniquement avec la suggestion.
'''
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()

    def suggest_gpu_directive(self, issue):
        prompt = f'''
Tu es un expert en parallélisation Fortran GPU (OpenACC).
Pour l'issue suivante : {issue['debt_type']} aux lignes {issue['line_range']},
propose la directive GPU adaptée. Répond uniquement avec la ligne de directive.
'''
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message["content"].strip()
