# Roo Code - Autocoder

Roo Code demonstrates an innovative approach to boosting developer productivity by leveraging cloud-hosted LLMs (Large Language Models) to assist coding tasks. Cloud-hosted LLMs like Claude Sonnet 3.7 are showing promising results with their context awareness and ability to insert useful code, though they come at a relatively high cost, making them economically questionable for exploratory or hobbyist tasks.

With Roo Code, you have the flexibility to choose the backend for your LLM, which opens possibilities for hosting also locally on consumer hardware. This is appealing as it provides it offers alleviates privacy concerns, and utilizes local hardwaere instead of paying a high price for every interaction.

But how well does it work? Can you get more productive? Or is it just hype? That's what I'm trying to explore and describe in this article.

## Landscape

Nearly every week brings new, groundbreaking innovations to the public space. The AI models that were turning the world upside down last week are replaced by even more capable models—sometimes for free—the following week. Models like GPT-40, Deepseek R1, Grok, and others continue to emerge. In this rapidly changing environment, it's hard to stay on top of the latest developments.

However, one thing remains true: higher intelligence is not useful if it cannot be applied. Unless we reliably solve the applicability problem of AI within practical fields, AI will remain a fascinating dream rather than a reality.

The world is starting to understand that there likely won't be a single LLM that can solve everything for humanity. Instead, numerous small models, carefully trained for specific purposes and well-orchestrated to fulfill difficult or tedious but specific tasks in the real world, are emerging. The industry is developing protocols like MCP to allow the creation of specific AI agents that can be combined into larger autonomous workflows. This approach, dubbed "Agentic AI" by technology leaders at Nvidia and Microsoft, makes training smaller, more specialized models manageable and economically feasible on smaller budgets. It also confines advances and regressions in specific areas, allowing controlled advancements at manageable quality.

## Auto-coders and Coding Assistants

The discipline of a software engineer can be seen as one such specific area where AI agents can become useful. In this field, we see several promising products emerging, such as Copilot, Windrurf, Continue, or Roo Code (formerly Roo Cline).

Coding assistants like Copilot, which act more like chat interfaces and provide very capable auto-completion, are rapidly becoming common in a software engineer's toolbox for rapid code analysis, development, review, and debugging.

The other category of tools, referred to here as auto-coders, can autonomously complete coding tasks. The engineer formulates a specific task for the autocoder, who then executes the task and provides the completed changes for refinement and approval.

Auto-coding requires much stronger context awareness and higher precision so that a coding task can be carried out to completion. There are many YouTube videos showing autocoders writing a complete Tetris game from scratch within minutes, giving rise to a new paradigm called Vibe Coding.

However, more often than not, software engineers work with large, existing codebases where years of thought have gone into producing the desired outcome concerning functionality, performance, security, robustness, or maintainability. To autonomously complete a task satisfactorily, the autocoder needs to infer from existing code the structure, intent, and quality aspects before coming up with a reasonable plan for making changes.

As LLMs are stateless, autocoders must manage context, iteratively refine their understanding of the context, call tools for knowledge acquisition, and apply changes to the actual codebase. This increased complexity sets it apart from coding assistants and puts more specific requirements on the LLM's capabilities. In other words, the LLM needs specific training, fine-tuning, and extraordinary precision when executing tasks. Particularly, precision may be a challenge with running LLMs on consumer hardware.

But can it be done? And if so, what tasks can it do?

## Why Roo Code?

Roo Code is well-established in the community as a super accessible plugin for VSCode that integrates seamlessly into VSCode workflows, such as applying code changes, executing commands directly in the VSCode terminal, integrating with source control, working with diffs, etc.

It can be configured to use local LLMs and is open-source, allowing us to explore why something works or doesn't and fix it if necessary. 

What stands out most is that Roo Code lets you define custom roles in the context of the codebase you're working on—e.g., for coding, as an architect, for asking questions to build understanding. Choosing a role affects Roo Code's behavior, making it feel like you're working with a colleague specializing in a particular area. You can ask for understanding, get architectural considerations, and have the coder implement changes to perfection.

While default roles do a decent job, you can also create your own—e.g., as a helpful reviewer or tech writer—and specialize them further to focus on specific quality aspects like robustness, performance, or security—important for real-world tasks. Additionally, Roo Code has an active community with frequent discussions on Discord and regular releases, lowering the time needed between ideation and release of new versions.

## Why Doing This?

With 30 years of professional experience in the software industry under constant technological change, one thing has been persistent: refactoring production systems is hard. It challenges engineers to their maximum capacity. Now comes AI, promising a truly helpful hand. It's a machine that never forgets and should theoretically be the perfect companion for tedious tasks or those where the human mind struggles to maintain complete oversight. The machine never has a bad day and is always ready to help. The ultimate Swiss Army knife: code that can assist with everything from speeding up learning, remembering missing semicolons, running through millions of lines of log files, or testing out new strategies — while you're having lunch or thinking about the next great thing to do. I think a good autocoder can truly a wonderful thing, and improve a software engineer's quality of life.

At the same time, it's not just about the end-state. With rapid development, what the end-state should be is no longer clear. Instead, it's more about learning about the inner workings of tools like autocoders and using them for exploratory purposes—even seemingly silly ideas—without having to think about monetary footprints for every interaction, which can really take the fun out of it.

Lastly, resource constraints tend to accelerate learning as things don't always work out as expected. You're presented with one challenge at a time, forcing you to understand the inner workings and consider different approaches.

## The Setup

For this work, I rely entirely on pre-owned hardware components purchased over the years or for this specific purpose. It's not the most modern setup but definitely high-end for consumer PCs. My configuration includes a previous-gen AMD Ryzen 7 CPU, 128GB DDR4 RAM, 2TB NVMe storage, and two GPUs on PCIe 4: RTX 3090 TI and RTX 3060, giving me a total of 36GB of VRAM to work with.

For software, I use Unraid OS because it's lightweight and provides painless support for containerization and virtualization with GPU passthrough. Initially, I ran Windows and Ubuntu on Unraid using QEMU for exploring Ollama, llama.cpp, Pinokio, LocalAI. But as I settled on the Ollama idea, I ditched the VMs and run everything in containers on Unraid. This keeps it lightweight and makes it all repeatable on other free platforms like Docker Compose or Kubernetes.

Roo Code runs on a Mac but with remote development on Unraid. Essentially, no software is installed on the Mac—other than VSCode—all workloads and AI-related processing are delegated to my Unraid server.

## How Auto-coders Work

Debugging VSCode and the Roo Code plugin to understand their inner workings can be daunting due to a lot of code running unrelated to LLM interactions. The way I found useful for building understanding was enabling debug logging on Ollama and monitoring HTTP traffic between Roo Code and Ollama.

As mentioned earlier, solving complex tasks requires numerous, iterative LLM interactions. Since LLMs are stateless and don't have access to storage or tool calling, everything the LLM needs to know to perform the next task is included and sent in every single call to the LLM.

In Roo Code, such calls range from 200KB upwards, even if you just ask the model to explain a few lines of code. With each iteration, the message grows in size as more information is collected, thus increasing the context.

The message to the LLM, or prompt, contains everything: for example, the system prompt explaining how the model should behave and format responses— including what tools it can request to be called. The user prompt includes your request to the LLM, the contents of any referenced files, and all previous interactions with the LLM so that it can decide what to do next without starting from scratch.

The reply from the LLM, or assistant prompt, is streamed back to Roo Code, one token at a time. Once Roo Code has received all tokens, it parses the message and decides what to do next. The LLM may request a tool call, such as read_file, if it thinks more information contained in a file wasn't provided before. It might ask for changes to be made to a file or a folder to be created. Roo will then ask you whether the request is reasonable and execute it if approved—or configured for auto-approval. The result of the tool call will then be embedded into the context for the next call to the LLM.

As Roo Code supports nearly infinite combinations of LLMs and operating systems with specific terminal capabilities, some tool calls may fail due to wrong assumptions by the LLM, lack of specificity, or vague system prompts in Roo Code. To handle these cases, Roo Code has built-in checks that evaluate the LLM response and send corrective instructions back.

The less your model, operating system, system prompts, and hardware capabilities are aligned, the more you will observe faults during task execution and have Roo Code circle back to the LLM for re-evaluation. Eventually, the LLM and Roo Code accumulate enough knowledge in the context and agree on making code changes, completing the task set up for it, and presenting the output.

Roo Code wraps things up and leaves it up to you to do a code review and submit the altered code to the source code repository. You're now ready for the next task.

## Here's What Is Hard

Simple tasks like "remove all unreferenced private functions in class XYZ" are easy, require few iterations, and are likely to succeed. Even intellectually difficult tasks like "improve my algorithm in function XYZ for performance at the expense of memory" can be carried out with ease—given that your LLM is trained for this.

When it comes to more involved tasks, such as "break down class XYZ for maintainability and factor functionality ABC out into a separate class," the LLM won't have all necessary information in one codebase. It needs to create new files, gather more information about where the class is instantiated from, and what functions are publicly used. The LLM and Roo Code will work iteratively to make step-by-step plans, gather information, create file structures, etc., building up a large context passed back and forth.

Two interesting things happen:

- Minor decisions in early iterations become hard facts as the evolving context builds around everything learned. Incorrect assumptions or imprecise formulations amplify subsequent iterations, standing indistinctly in the context for the duration of the entire task. If the LLM is not trained or fine-tuned for precise interactions with the task at hand, the likelihood of wandering off into a wrong decision path and producing unusable code increases. Roo Code won't detect this, leaving you to wonder what happened, undo all changes, and try again with a simpler task.

- With increasing iterations, the physical size of the context grows large and eventually outgrows the model's context limitations. When that happens, the model either truncates its response or becomes very loose when considering the provided context. In the best case, Roo Code gets upset and decides to quit the task. In the worst case, Roo Code keeps trying to get the model back on track but makes the matter with context size worse.

In essence, simpler tasks have a higher chance of success, but you gain less from the Roo Code setup. To benefit most, find the best possible combination of model, configuration, system prompt, and hardware capabilities for your task. If all parameters are fixed, break down tasks for Roo Code into smaller, more manageable pieces.

This process can cause a lot of frustration as there is no easy fix. Solutions are individual, and not much guidance is available on the internet.

And this is probably exactly why hosted solutions like Claude Sonnet 3.7 are attractive—despite their high costs. Their configuration is consistent, models are fine-tuned with great precision, Roo Code has system prompts tested with it, and backing hardware supports extremely large context. Essentially, you get the best possible setup—at an unfavorable price point.

## Scaling Up from Your Experience

Once you master optimization on your local setup, you have the knowledge needed to scale up using rented GPUs through providers like runpod.io. There, you can host your Ollama backend with your configuration on powerful GPUs for as little as half a dollar per hour and have tokens pass back and forth at a flat rate. When done, just shut it down so you don't pay overnight.

Roo Code and the knowledge about local setup will definitely play well in that environment.

## Setting the Ambitions

### Speed vs Size

Most models can work purely on CPU and off-load to GPU for faster processing. This makes it virtually possible to run the largest models with complex tasks on cheap hardware—with enough system RAM and disk space. However, for reasonable speeds, the model and context should ideally fit entirely onto your GPU, reducing VRAM swapping. Running inference tasks on the GPU instead of a combination of CPU and GPU literally makes a speed difference of factors tens or hundreds.

For practical application with iterative refactoring of code, it is highly advisable to run 100% on GPU. When loading the configured model in Ollama, check:

```bash
ollama ps
```

To what extent your configuration uses the GPU.

### How Many Billions of Parameters?

Models come in sizes determined during training, often designated in their names or tags: 230b, 70b, 24b, 14b, 8b, 3b, 1.5b, etc. The more parameters in the model, the more nuanced it can be.

My hardware and 36GB VRAM max out with models between 8b and 14b—depending on context size used. I can run 24b models, but available space for context becomes impractically small.

So, go as large as your setup permits in terms of parameters.

### Training

When choosing a model, it's best if the billions of parameters were used during training for coding tasks. The majority of the training should be done on computerized languages (programming languages), with some on natural language so the LLM can understand requests and perform programming accordingly.

Models designated as 'instruct' or 'coder' are fine-tuned for following automated instructions or specific coding tasks, but this is not a reliable measure. You need to read up on what models work well.

Popular models include Mistral, Qwen2.5, Phi4, CodeLlama, Deepseek-Coder, Falcon, and more. Some models are distilled from larger, general-purpose models, allowing smaller, more specific models to have similar or even better capabilities using less hardware.

Finding a model that suits your task is your own challenge. For my part, I seem to have best results using Qwen2.5-derived models.

### Quantization

Next to the context, most VRAM utilization of a loaded model comes from trained parameters (model weights) stored in multidimensional arrays (tensors). Every parameter is represented by a datatype—most commonly 32-bit as fp32 during training. 64-bit representations offer better precision but are extremely uncommon outside scientific realms.

Here's the problem: if you choose a 24b model with training that fits your needs and run it at 32-bit precision, your model alone will consume `24b * 32bit = 96GB` of VRAM. This doesn't even include memory required for context. That amount of VRAM hardly counts as consumer-level hardware—you can rent them through providers like runpod.io at hourly rates.

Here's where quantization comes in: it is the conversion of parameters into smaller, lower-precision data types. Common representations are 16-bit (fp16 or bf16) at half precision—therefore requiring half the space.

With my 36GB of VRAM, I need to go even lower, for example to 8 bits (Q8), further down to Q6, Q5, or Q4. Reduction in data precision affects model precision, and at too low precision, the model will start hallucinating easily, turning refactoring tasks into comedy.

In my case, with 36GB of VRAM, I see reasonable results with Q8 quantization of a 14b model on Qwen training.

### Context Size

Context size is the last but probably most important thing to consider. While easy to imagine, it's likely the least trivial to get right. Here's some explanation:

Context size refers to the number of tokens a model can process at a given point in time—think of it as active memory for processing. A token is a sequence of bytes identified during model training—typically a word, stem, or punctuation. The larger the context, the more complex the task you can express.

In the use case of Roo Code, messages sent to the LLM can become very large due to reasons discussed earlier. On low-context configurations, there might not be enough space for the result message, derailing any refactoring attempt.

The challenge is that while Ollama lets you freely choose context size to suit your needs—or max out your hardware—your LLM may not perform reliably on larger context sizes. If the model itself was trained with a particular context size, results can be arbitrary when exceeding that trained size.

I've tried with varying results going by 1.5 to 2.0 times larger than the trained context size. Things then become notably more inconsistent, and LLM and Roo Code seem to wander in circles.

But how do you identify the trained context size so you don't overshoot? Unfortunately, this is only known during training and can't be seen on the model except when attached as metadata. You can see this in the Ollama log when loading the model.

Here's an example of the Unsloth/phi4 model:

```log
llm_load_print_meta: n_ctx_train      = 16384
```

The model was trained with a 16k context. Expanding that to 128k context will likely not make you happy.

Moreover, some models state support for larger contexts—although they have been trained for smaller ones. They apply a technique called role scaling, essentially extrapolating at the expense of precision. Depending on the algorithm, they scale linearly or more precisely in the lower end and less precisely in the higher end of the context.

You can also see such properties in the Ollama log (with my annotations):

```log
llama.context_length	16,384	# Base trained context length
llama.rope.dimension_count	128	# RoPE position encoding size
llama.rope.scaling.type	linear	# Uses linear RoPE scaling for extended context
llama.rope.scaling.factor	4.0	# Suggests it was extrapolated to 4× its trained length (16K × 4 = 64K)
```

So, what's the learning here? Consult the model's documentation and verify behavior in the Ollama log for setting the initial context size. Bumping the context size up to max is likely not giving you the desired result.

In my setup, on 36GB VRAM, I seem to find a good lee-way with the Qwen model, traind for the 1M context size, but sizing it down to 42k for fitting into my hardware. The logs are reflecting this along those lines.

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

## Where to Get the Models From?

Naturally, as we're working with Ollama, the natural starting point is on http://ollama.org. However, far more combinations of training methods, quantizations, fine-tunings, distillations can be found on Huggingface at http://hugginface.co. If you are new to this, start out with Ollama. It's well-contained. Once you know what you need, move on to Huggingface. This is the Pandora's box—once you open it, there's no turning back.

For keeping track of my models, I create modelfiles in a folder, one for each model I'm testing with. Then I have a small script that rebuilds all models whenever I make a change. Here are a few examples:

### Example with Qwen from Huggingface.co:
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

### Example with Deepseek R1 from Ollama.org:
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
ollama create mymodel -f mymodelfile
```

The reason I don't supply the context size during Ollama startup is that modelfiles support many other parameters, allow comments, and integrate easier with a source control workflow.

If you do this at scale, you may find joy in my script for rebuilding all model files in the current folder. The model files must be named according to the following standard: [mymodelname]-[mymodeltag].modelfile

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

## Testing the Task

I have installed Roo Code, pointed it at my Ollama instance, with the following model configuration:

```modelfile
FROM hf.co/lmstudio-community/Qwen2.5-14B-Instruct-1M-GGUF:Q8_0
PARAMETER num_ctx 43008
```

This maxes out my 36GB of VRAM almost entirely, and gives me the joy for a few simple tasks.

I use the deepmeepbeep's YeEGP repo for testing, as I still have it lingering around from an earlier experiment: https://github.com/deepbeepmeep

And I run Roo Code over it using the following prompt for the coding role:

```quote
I want you to improve maintainability of the Python code in @/inference/gradio_server.py. Organize the imports, and factor the args parsing out into a new file utils/args.py. Add short documentation on every function to indicate the purpose of the function.
```

I could definitely break this down into smaller tasks and make it more specific. But I want to see if this task can be handled. It seems like a fair, real-world thing to do.

The task takes several iterations—some intentional, others corrective due to model imprecision—and x minutes to complete. The resulting code is useful and works as intended.

## Conclusion

Getting to this point has not been straightforward. Luckily, I am equipped with a healthy amount of curiosity and a high threshold for pain. I could have chosen an easier setup.

Though, I would never send the code as a pull request, as the model does not pay much attention to existing structures and formatting, and you see code blocks flipping around.

All in all, I think I have a setup that works. Through fine-tuning of the system prompt and more carefully wording my instructions, I could probably reach a satisfying state. I am impressed.

However, considering the amount of effort it has taken me to make this particular setup and the speed at which new models evolve, the setup likely will need revision. I'm not sure whether I can wholeheartedly recommend that to everyone.

In my case, where I like to tinker with tools for recreational purposes, maintaining that setup seems fine. But if I was in it purely for productivity boosts, I am not sure whether I could justify the hardware and time investments at this point.

Taking the hit and using Roo Code with a hosted provider may get you up and running faster and more predictably.
