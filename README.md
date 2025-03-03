# Roo Code - A Massive Productivity Boost

## Motivation

Roo Code boosts developer productivity by leveraging cloud-hosted LLMs (Large Language Models). Cloud-hosted LLMs like Claude 3.7 Sonnet and OpenAI's GPT-4o demonstrate remarkable context awareness and the ability to solve complex coding tasks.

Here is a recording of two real-world refactoring tasks with Roo Code against OpenAI GPT-4o (with deliberately loose instructions). Quite cool:

[![Watch the video](media/roo-openai.png)](media/roo-openai.mp4)

However, with cloud-hosted backends comes a steep cost, as you pay per token. In complex refactoring, you quickly end up with millions of tokens used. This takes the fun out of it quite a bit, e.g., for recreational coding projects.

Luckily, with Roo Code, you can choose the backend and also run against locally hosted LLMs using Ollama. This can save you tons of money if you choose to run it on capable consumer hardware—like your son's gaming PC while he is sweating in school (not that I would ever do that 🫣).

But how well does it work? Why is not everybody doing it this way? As I was looking for answers on the internet, I found a lot of information about problems but not much useful guidance for how to overcome them.

GusoCoder published an encouraging video a few weeks ago, which triggered me to give it a spin on my own hardware. I wanted to understand what was going wrong and what we potentially could do to fix it.

[![Watch the YouTube video](media/local-llm.png)](https://www.youtube.com/watch?v=7sgSBLIb0ho)

In this article, I describe my experiences in the hope of inspiring someone else to also share their experiences—or at least to save someone time trying to get their heads around it.

### For the Impatient

Yes, it works! But you need to keep your ambitions a bit lower, as the local setup cannot compete with the cloud-hosted model in speed and quality. Though, if you are willing to put in the time to tune and optimize your setup, you can achieve good results.

Plus... there is an option to choose a hybrid approach, such as hosting Ollama yourself on services like runpod.io, which provide good hourly rates for powerful GPUs. This lets you go absolutely crazy for a few hours without worrying about cost every time you press a button.

### Landscape in February 2025

Nearly every week brings new, groundbreaking innovations to the public AI space. The AI models that were turning the world upside down last week are replaced by even more capable models the week after—sometimes even for free. Models like GPT-4o, Deepseek R1, Grok, Mercury, and others continue to emerge. In this rapidly changing environment, it seems super hard to stay on top of the latest developments.

While ever-higher intelligence is fascinating, it is hardly useful for the masses if it cannot be applied in practical fields. Unless we reliably solve the applicability problem, AI will remain a fascinating dream rather than a reality for most of us.

The world is recognizing that no single LLM can solve everything. Instead, smaller, specialized models are emerging, each trained for specific tasks and orchestrated to handle complex or tedious work efficiently. Industry protocols like MCP enable interoperability within a broader AI ecosystem.

This approach offers key advantages: smaller models are easier and more cost-effective to train, promote reusability, and isolate both advancements and regressions within specific domains. This ensures controlled progress while maintaining quality across the ecosystem.

It seems that we are at a turning point towards decentralization and democratization of AI. Something that will definitely help the broader adoption into real-world tasks.

### Auto-Coders and Coding Assistants

The software engineering discipline fits very well the description of such a specific area with real-world tasks. In this field, we see several promising products emerging, such as Copilot, Windrurf, Continue, or Roo Code (formerly Roo Cline), making use of Large Language Models (LLMs).

On one hand, there are coding assistants like Copilot, which act more like chat interfaces and provide capable auto-completion. These are rapidly becoming common in a software engineer's toolbox for accelerated code analysis, development, review, and debugging.

The category of auto-coders, on the other hand, can autonomously complete coding tasks. The engineer formulates a specific task for the auto-coder, which then executes the task and provides the completed changes for refinement and approval. Auto-coding requires much stronger context awareness and higher precision of the model so that a coding task can be carried out to completion.

```mermaid
%%{ init: { 'theme': 'base', 'themeVariables': { 'primaryColor': '#f0f0f0', 'edgeLabelBackground':'#ffffff', 'fontSize':'14px' } } }%%
flowchart TD
    subgraph "AI Coding Assistants"
        A[Coding Assistants] -->|Depend on| B(Small, Fast Models)
    end
    
    subgraph "AI Auto-Coders"
        C[Auto-Coders] -->|Depend on| D(Large, Powerful Models)
    end
    
    style A fill:#007acc,stroke:#000,color:#fff
    style B fill:#89c4f4,stroke:#000
    style C fill:#008000,stroke:#000,color:#fff
    style D fill:#90ee90,stroke:#000
```

More often than not, software engineers work with existing codebases where years of thought have gone into producing the desired outcome concerning functionality, performance, security, robustness, or maintainability. To autonomously complete a refactoring task satisfactorily, the auto-coder needs to infer from existing code the structure, intent, and quality aspects before coming up with a reasonable plan for making changes.

Doing this well requires the build-up of large context for achieving the desired result. Auto-coders must manage this context and help the model to iteratively refine the understanding, call tools for knowledge acquisition, and apply changes to the actual codebase. This more involved workflow is what sets auto-coders apart from simple coding assistants.

### Appreciation for Roo Code

Roo Code is well-established in the community as a super accessible auto-coder that integrates seamlessly into VSCode workflows, such as applying code changes, executing commands directly in the VSCode terminal, integrating with source control, working with diffs, etc. It can be configured to use local models using Ollama and is open-source, allowing us to explore why something works or doesn't and fix it if necessary.

The popularity of Roo Code can be seen e.g. on the Open [Router web site](https://openrouter.ai), which shows a ranked list of applications that are used on their platform. Roo Code ranks behind Cline, with a large gap to SillyTavern: 

![Open Router Ranking](media/open-router-roo-code.png)

Roo Code is a fork of Cline, but has added support for modes, e.g. for coding, as an architect, for asking questions to build understanding. Choosing a mode affects Roo Code's behavior, making it feel like you're working with a colleague specializing in a particular area. You can ask for understanding, get architectural considerations, and have the coder implement changes to perfection. While default modes do a good job, you can also create custom modes — e.g., as a helpful reviewer or tech writer—and specialize them further to focus on specific quality aspects like robustness, performance, or security—important for real-world tasks. Custom modes make it super extendable.

```mermaid
graph TD;
    A[Roo Code Modes] -->|Writing & modifying code| B[Code Mode];
    A -->|Structuring & designing| C[Architect Mode];
    A -->|Debugging & fixing| D[Debug Mode];
    A -->|Discuss & understand| E[Ask Mode];
    A -->|User-defined behavior| F[Custom Modes];
    
    F -->|Example: Generate documentation| G[DocGen Mode];
```

Moreover, Roo Code has an active community with frequent discussions on Discord and regular releases, lowering the time needed between ideation and release of new versions.

This combination makes Roo Code stand out to me as a versatile tool that I believe can follow my emerging needs and keep up in the fast-changing environment.

## Identifying the Problem

### Fast, Cheap, and at Top Quality

If you choose to run your LLM on consumer hardware, as opposed to cloud-hosted services, you run into constraints that you will need to balance. If you wouldn't care that a task completes first after 14 hours, you can run the most powerful LLM against your codebase. But if you want a more interactive way of working, you will need to sacrifice model capabilities instead.

As auto-coders rely heavily on context management, everything in this article will revolve somehow around this.

But first, let's have a look at how Roo Code works.

### So, How Does Roo Code Work?

Debugging VSCode and the Roo Code plugin to understand their inner workings can be daunting due to a lot of code running unrelated to LLM interactions. If you want to explore the inner workings, a better way may be to enable debug logging on Ollama and monitor HTTP traffic between Roo Code and Ollama.

You will see that LLMs are inherently stateless. They have no access to storage or tools, and every interaction with the LLMs is accompanied with complete context of what Roo Code is about to do. Context starts typically out small in the beginning of the task, but grows larger as Roo Code iteratively progresses the work together with the LLM. Solving complex tasks requires many iterations, which leads to large context sizes.

In Roo Code, I see calls to the LLM range from 200kb per call upwards, even if I ask the model to explain a few lines of code. With each iteration, the message grows in size as more information is added.

The message to the LLM contains everything:

- The system prompt explaining how the model should behave and format responses — including what tools it can request to be called.
- The user prompt includes your request to the LLM.
- The contents of any referenced files.
- A history of all previous interactions with the LLM so that it can decide what to do next without repeating itself.

For illustration of the prompts and iterations, the structure of the message to the LLM may look something like this (shortened version):

```json
{
    "model": "olilanz:qwen-v2.5",
    "messages": [
        {
            "role": "system",
            "content": "You are Roo, a highly skilled software engineer with ...[shortened]"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<task>\nI want you to improve maintainability of  ...[shortened]"
                },
                {
                    "type": "text",
                    "text": "<environment_details>\n# VSCode Visible Files\ ...[shortened]"
                }
            ]
        },
        {
            "role": "assistant",
            "content": "<thinking>\nTo improve the maintainability of the Python ...[shortened]"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "[read_file for 'inference/gradio_server.py>\n</read_file>'] Result:"
                },
                {
                    "type": "text",
                    "text": "The tool execution failed with the following error:\n ...[shortened]"
                },
                {
                    "type": "text",
                }
            ]
        },
        {
            "role": "assistant",
            "content": "It seems there was an issue with the file path in the  ...[shortened]"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "[read_file for 'inference/gradio_server.py'] Result:"
                },
                {
                    "type": "text",
                    "text": "  1 | import os\n  2 | import sys\n  3 | from  ...[shortened]"
                },
                {
                    "type": "text",
                    "text": "<environment_details>\n# VSCode Visible Files ...[shortened]"
                }
            ]
        }
    ],
    "temperature": 0,
    "stream": true
}
```

You see multiple user/assistant interactions - indicating that this is the message at the start of teh 3rd iteration. 

After the call cpletes, the reply from the LLM, or assistant prompt, is streamed back to Roo Code, one token at a time. Once Roo Code has received all tokens, it parses the message and decides what to do next. The LLM may request a tool call, such as read_file, if it thinks more information contained in a file wasn't provided before. It might ask for changes to be made to a file or a folder to be created. Roo will then ask you whether the request is reasonable and execute it if approved—or configured for auto-approval. The result of the tool call will then be embedded into the context for the next call to the LLM.

```mermaid
sequenceDiagram
    participant User
    participant RooCode
    participant LLM
    participant VSCode

    User->>RooCode: Send code base and task
    loop Until task is complete
        RooCode->>LLM: Send context for iteration
        LLM->>RooCode: Return context response
        RooCode->>VSCode: Execute commands (optional)
        VSCode->>RooCode: Return execution result
        RooCode->>User: Ask for more guidance (optional)
        User->>RooCode: Return guidance
    end
    RooCode->>User: Return updated code base
```

Some tool calls are prone to fail due to wrong assumptions by the LLM, lack of specificity in the task, or vague system prompts in Roo Code. After all, Roo Code supports many combinations of LLMs and operating systems, and it's system prompt may therefore not match perfectly your setup. To handle these cases, Roo Code has built-in checks that evaluate the LLM response and send corrective instructions when faults happen. In the example above, the read_file operation failed, after which Roo Code asks the model again to be more specific about the file path. 

The less your model, operating system, system prompts, and hardware capabilities are aligned, the more you will observe faults during task execution and have Roo Code circle back to the LLM for re-evaluation. Eventually, the LLM and Roo Code accumulate enough knowledge in the context and agree on making code changes, completing the task set up for it, and presenting the output.

Roo Code wraps things up and leaves it up to you to do a code review and submit the altered code to the source code repository. You're now ready for the next task.

If all goes well, Roo Code and the LLM conclude that the task was successfully completed, and return back to you to formulate a new task. The new task will start fresh with an empty history. 

### Here's What is chalanging with local LLMs

Simple tasks like "remove all unreferenced private functions in class XYZ" are easy, require few iterations, and are likely to succeed. Even intellectually difficult tasks like "improve my algorithm in function XYZ for performance at the expense of memory" can be carried out with ease — given that your LLM is trained for this. The task itself, the system prompt, as well as the complete context can be wrapped up and sent at once.

But when it comes to more involved tasks, such as "break down class XYZ for maintainability and factor functionality ABC out into a separate class", the LLM won't have all necessary information in one pass. It needs to create new files, gather more information about where the class is instantiated from, and what functions are publicly used. The LLM and Roo Code will work iteratively, make step-by-step plans, gather information, create file structures, etc., building up a large context passed back and forth.

Two interesting things now happen:

- Minor decisions in early iterations become hard facts as the evolving context builds around everything learned. Incorrect assumptions or imprecise formulations amplify in subsequent iterations, standing indistinctly in the context for the duration of the entire task. If the LLM is not capale of precise interactions with the task at hand, the likelihood of wandering off into a wrong decision path and producing unusable code increases. Roo Code won't likely detect this, leaving you to wonder what happened.

- With increasing number of iterations, the physical size of the context grows large and eventually outgrows the model's context limitations. When that happens, the model either truncates its response or becomes very loose when considering the provided context. In the best case, Roo Code gets upset and decides to quit the task after a few unsuccessful iterations. In worst case, Roo Code keeps trying to get the model back on track but makes the matter with context size worse inevery iteration.

In essence, simpler tasks have a higher chance of success. So, to benefit most, find the best possible combination of model, configuration, system prompt, and hardware capabilities for your task. If all parameters are defined, break down tasks for Roo Code into smaller, more manageable pieces.

Finding the right balance is difficult, and can cause a lot of frustration. One day it all seems to work, and in the next project you will not get anything useful out of it. There is no definitive answer for what works in all cases. Solutions are individual.

This is exactly why hosted solutions like Claude 3.7 Sonnet keep attracting customers — despite their high costs. Their configuration is consistent, models are fine-tuned with great precision, Roo Code has system prompts tested with it, and backing hardware supports extremely large context. Essentially, you get the best possible setup — at a steep price point - but it just works.


## Breaking the problem down, step by step

### Do you want speed or size for a coding task?

Most models can work purely on CPU and off-load to GPU for faster processing. This makes it possible to run the largest models with complex tasks on cheap hardware — with enough system RAM and disk space. However, for reasonable speeds, the model and context should ideally fit entirely onto your GPU, reducing VRAM swapping. Running inference tasks on the GPU instead of a combination of CPU and GPU literally makes a speed difference of factors tens or hundreds.

When loading the configured model in Ollama, you can use the following command to see to what extent Ollama runs your model on the GPU:

```bash
ollama ps
```

For practical application with iterative refactoring of code, it is highly advisable to run 100% on GPU, so you get your responses before you decide to wanter off and do something else insead. 

### How Many Billions of Parameters do you need?

Models come in sizes that a re defined during training, often designated in their names or tags as: 230b, 70b, 24b, 14b, 8b, 3b, 1.5b, etc. The more parameters in the model, the more nuanced it can be.

On my setup, model with 8b and 14b provide a good starting point. With 36GB of VRAM, there is then still ample of space left for context.

I can run 24b models, but the available space for context becomes impractically small.


### Focussed Training matters - remove the dead waight

When choosing a model, it's best if the training of the billions of parameters is dedicated to coding tasks. The majority of the training should be done on programming languages. General purpose models carry a lot of dead weight along, which is not useful for programming. Such as how to combine ingredients to make a perfect stew, or the long term effects of deforestation of rain forests. Try to avoid those for best results.

Models designated as 'instruct' or 'coder' indicate that they are fine-tuned for following automated instructions or specific coding tasks. They tend to be worth a closer look.

Popular models that seem to work well with coding include Mistral, Qwen2.5, Phi4, CodeLlama, Deepseek-Coder, and more. Some models are distilled from larger, general-purpose models, allowing them to have similar or even better capabilities using less hardware.

Finding a model that suits your task is your own challenge. For my part, I seem to have best results using Qwen2.5-derived models.

### Quantization: Save tons of space at the cost of a little precision

Next to the context, most VRAM utilization of a loaded model comes from trained parameters (aka model weights). They are stored in multidimensional arrays (aka tensors). Every parameter in VRAM is represented using a datatype — most commonly 32-bit as fp32 during training.

But for your local inference task, this may be a problem. Here's a quick calculation: if you choose a 24b model with training that fits your needs and run it at 32-bit precision, your model alone will consume `24b * 32bit = 96GB` of VRAM. That amount of VRAM hardly counts as consumer-level hardware - and you have not even included the memory for context yet.

This is where quantization comes in. It lets you reduce the memory footprint, so the model fits into your VRAM while also leaving speace for context. Quantization is the conversion of parameters into smaller, lower-precision data types.

Common quantization is to 16-bit (fp16 or bf16) at half precision — therefore requiring half the space. With low VRAM, you likely need to go even lower than that, for example to 8 bits (Q8), or even further down to Q6, Q5, or Q4. 

Though, reduction in data precision directly affects model precision, and at too low precision, the model will start hallucinating easily, turning refactoring tasks into comedy. Roo Code and the LLM will the start arguing. Imagine an old couple arguing at the dinner table - one has bad vision see, the other has bad hearing. 

![Old couple](media/old-couple.jpg)

In my case, with 36GB of VRAM, I see reasonable results with Q8 quantization of a 14b model on Qwen architecture and training.

### Learning: Increasing the context size can deteriorate the quality

Context size is the last but probably most important thing to consider. While easy to imagine, it's likely the least trivial to get right. Here's some explanation:

The contex size os measured in number of tokens. Tokens are sequences of bytes that were identified during model training — such as a word, stem, or punctuation. Typically only a few bytes long. 

When determining the optimal context size for your setup, you need to consider what the context is used for. Firstly, there is the part that is sent to the LLM. But secondly, there is also the LLM's response. So, if the first part becomes too large, the second part will be truncated. This renders the LLM unusable in the use case of Roo Code. 

As Ollama lets you freely define your preferred context size, there is a temptation to just crank that number up to fill up the VRAM. Unfortunately, not all models work reliably with large context sizes. If the model itself was trained with a smaller context size, results can be arbitrary when running the inference tasks with a significantly larger context size. The cpntext size that you configure your model with in Ollama needs to consider the model's original training size. 

When exceeding the trained context size by a factor larger than 1.5 to 2.0, results may start to degrade again. And a high parameter model starts producing poor reults. Things then become notably more inconsistent, and LLM and Roo Code start having a heard time to make progress together. Again like the old couple from before.

### How to identify the trained context size?

Unfortunately, the trained context size is only known during the training of the model, and can't be seen on the model itself - except when the trainers kindly attach this information as metadata to the model. In that case you can see this in the Ollama log when loading the model.

Here's an example of the logs when Ollama loads the Unsloth/phi4 model:

```log
llm_load_print_meta: n_ctx_train      = 16384
```

The model was trained with a 16k context only. It may perform poorly with a 64k context size or above. If you observe this, then that might be the reason.

Moreover, some models state support for larger contexts — although they have been trained for smaller ones. They apply a technique called RoPE scaling (Rotary Position Embeddings), essentially extrapolating at the expense of precision. Depending on the algorithm, they scale linearly or more precisely in the lower end of the context, and less precisely in the higher end of the context.

You also ovserve the use of ROPE scaling in the Ollama log (with my annotations):

```log
llama.context_length	16,384	  # Base trained context length
llama.rope.dimension_count	128	  # RoPE position encoding size
llama.rope.scaling.type	linear	  # Uses linear RoPE scaling for extended context
llama.rope.scaling.factor	4.0	  # Suggests it was extrapolated to 4× its trained length (16K × 4 = 64K)
```

So, what's the learning here? You need to consult the model's documentation and verify behavior in the Ollama log for setting the initial context size. Bumping the context size up to max is prone to not giving you the desired result.

In my setup, on 36GB VRAM, I seem to find a good lee-way with the Qwen model, trained with 1M context size, but sizing it down to 42k for fitting into my VRAM. The logs are reflecting this along those lines.

```log
llama_new_context_with_model: n_seq_max     = 1
llama_new_context_with_model: n_ctx         = 43008
llama_new_context_with_model: n_ctx_per_seq = 43008
llama_new_context_with_model: n_batch       = 512
llama_new_context_with_model: n_ubatch      = 512
llama_new_context_with_model: flash_attn    = 0
llama_new_context_with_model: freq_base     = 10000000.0
llama_new_context_with_model: freq_scale    = 1
llama_new_context_with_model: n_ctx_per_seq (43008) < n_ctx_train (1010000) -- the full capacity of the model will not be utilized
```
### So, what is the solution?

If your setup works with Roo Code and your local LLM, you are in luck. More likely, it will work for some things, while it will disappoint for others. As the behavior is not really deterministic, you may need to try a few times (belss the folks who made Roo Code's new Ckeck-point feature):

So if your setup struggles, cosider this: 

- Changing to another model may make sense. But it may well be that the model is just poorly configured. So, if a model seems to work for someone else, but not for you, there might be a good explanation. 

- Check if Ollama is using your hardware efficiently. You don't want your VRAM to sit idle - or work on other tasks - like fancy animations of your desktop environment. 

- Bump up parameter size, quantization, and context size to fill up your available VRAM. Don't drop quantization below 8 if you can avoid it. Also keep the number of parameters as high as your available VRAM permits.

- Consider carefully the context size. Have you been too optimistic, and cranked it all up? Check how the model was trained, and if it performs safely with extended context. 

If all that is right, and you still find too much trouble, then it's about the prompting. 

- Be more specific when asking the Roo Code to do something. Do it in small steps. Keep the tasks confined in a single file, or a single function, so you don't have Roo Code iterate unnecessarily.

And if you still want more, consider the off-topics - like renting a GPU at runpod.io, or find a second GPU to cramp into your box. While this is not as efficient as having one large GPU with tons of VRAM, it is still a viable, cost effective approach to consider.


## Testing in the wild

### My setup and testing ground

For this investigation, I rely on pre-owned hardware components collected over the years. It's not the most modern setup, but beefy enough for the task, and still representative for a high-end consumer PC. My configuration includes a previous-gen AMD Ryzen 7 CPU, 128GB DDR4 RAM, 2TB NVMe storage, and an RTX 3090 TI on 16xPCIe4. I threw in an additional RTX 3060 on 4xPCIe4, giving me a total of 36GB of VRAM to work with.

<img src="media/hardware.png" alt="hardware" width="800" height="600">

For software, I use Unraid OS because it's lightweight and provides painless support for containerization and virtualization with GPU passthrough. Initially, I ran Windows and Ubuntu on Unraid using QEMU for exploring Ollama, llama.cpp, Pinokio, LocalAI. 

<img src="media/unraid.png" alt="unraid" width="800" height="600">

But as I settled on the Ollama idea, I ditched the VMs and run everything in containers on Unraid. This keeps it lightweight and makes it all repeatable on other free platforms like Docker Compose or Kubernetes. It also saves me a bit of valuable VRAM as Unraid doesn't have a desktop environment with GPU support.

Roo Code on the other hand runs on a Mac but with remote development on Unraid. Essentially, no software is installed on the Mac—other than VSCode—all workloads and AI-related processing are delegated to my Unraid server.

### The coding task

I have installed Roo Code, pointed it at my Ollama instance, with the following model configuration:

```modelfile
FROM hf.co/lmstudio-community/Qwen2.5-14B-Instruct-1M-GGUF:Q8_0
PARAMETER num_ctx 43008
```

This maxes out my 36GB of VRAM almost entirely, and gives me the joy for a few simple tasks.

<img src="media/nvtop.png" alt="nvtop" width="800" height="600">

I use the deepmeepbeep's YeEGP repo for testing, as I still have it lingering around from an earlier experiment: https://github.com/deepbeepmeep

And I run Roo Code over it using the following prompt for the coding role:

```quote
I want you to improve maintainability of the Python code in @/inference/gradio_server.py. Organize the imports, and factor the args parsing out into a new file utils/args.py. Add short documentation on every function to indicate the purpose of the function.
```

I could definitely break this down into smaller tasks and make it more specific. But I want to see if this task can be handled. It seems like a fair, real-world thing to do.

The task takes several iterations—some intentional, others corrective due to model imprecision—and x minutes to complete. The resulting code is useful and works as intended.

### Testing different configuration efficiently

The calibration process is time consuming as it requires a lot of trial and error. Here is a short description of the approach I have chosen.

For keeping track of my models, I create modelfiles in a folder, one for each model I'm testing with. Modelfiles provide the option to work many more paramaters, write comments and store everything safely in version control. In addition to that, I use a small script to rebuild all models when I have made changes. 

Example modelfile with Qwen from Huggingface.co:
```modelfile
# Very promising! Few minor mis-steps, but able to recover and continue.
# 56k was slow but stable. 48k is fast, but crashed Ollama worker process.
# Still testing.

# Prompt:
# I want you to improve maintainability of the Python code in @/inference/gradio_server.py . Organize the imports, and factor the args parsing out into a new file ultils/args.py. Add short documentation on every function to indicate the purpose of the function.

FROM hf.co/lmstudio-community/Qwen2.5-14B-Instruct-1M-GGUF:Q8_0
# Supports up to 1M context size.
# 40GB VRAM at 56k context size.
PARAMETER num_ctx 57344
# 36GB VRAM at 48k context size. But crashes!?
PARAMETER num_ctx 49125
# 34GB VRAM at 42k context size. But crashes!?
PARAMETER num_ctx 43008

# PARAMETER num_ctx 32768
# PARAMETER num_ctx 65536
```

Example modelfile with Deepseek R1 from Ollama.org:
```modelfile
# Very promising. Though keeps asking without "user_question". 
# The code is then written into the console instead of the file.
# As a result, the files were almost empty.

# The model has an engineered context size max context size of 128k. However, it's not sure how well it will perform on the large end.

# With context size 32k, the model uses 36GB of VRAM
FROM tom_himanen/deepseek-r1-roo-cline-tools:14b
PARAMETER num_ctx 57344

# FROM tom_himanen/deepseek-r1-roo-cline-tools:8b
```

You can get Ollama to download all dependencies and create your own model based on the base model and configured context size:

```bash
ollama create [mymodel] -f [mymodelfile]
```

If you do this at scale, you may find joy in my little script for rebuilding all model files in the current folder. The model files must be named according to the following standard: [mymodelname]-[mymodeltag].modelfile

```bash
#! /usr/bin/bash

# Get the directory of the script
script_dir="$(dirname "$0")"

rebuild_model() {
    local filename="$1"
    local base_name="$(basename "${filename%.*}")"
    local model="${base_name%%-*}"
    local tag="${base_name#*-}"

    echo "Rebuilding model $model with tag $tag from file $filename..."

    ollama rm "$model:$tag"
    ollama create "$model:$tag" -f "$filename"
}

# Call rebuild_model for every *.modelfile in the script directory
for modelfile in "$script_dir"/*.modelfile; do
    if [ -e "$modelfile" ]; then
        rebuild_model "$modelfile"
    fi
}
```

## Conclusion

Getting to this point has not been straightforward. Luckily, I am equipped with a healthy amount of curiosity and a high threshold for pain. I could have chosen an easier setup.

Though, I would never send the code as a pull request, as the model does not pay much attention to existing structures and formatting, and you see code blocks flipping around.

All in all, I think I have a setup that works. Through fine-tuning of the system prompt and more carefully wording my instructions, I could probably reach a satisfying state. I am impressed.

However, considering the amount of effort it has taken me to make this particular setup and the speed at which new models evolve, the setup likely will need revision. I'm not sure whether I can wholeheartedly recommend that to everyone.

In my case, where I like to tinker with tools for recreational purposes, maintaining that setup seems fine. But if I was in it purely for productivity boosts, I am not sure whether I could justify the hardware and time investments at this point.

Taking the hit and using Roo Code with a hosted provider may get you up and running faster and more predictably.
