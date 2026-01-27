<?php

/**
 * @file
 * Ollama adapter for accessing locally-hosted AI models.
 *
 * Ollama provides OpenAI-compatible API endpoints, making integration seamless.
 * This adapter connects to a local or remote Ollama server.
 *
 * @see https://docs.ollama.com/api/openai-compatibility
 */

// Prefer Composer Manager's autoloader when available; fall back to module vendor.
$__openai_sdk_source =& backdrop_static('openai_sdk_source');
if (module_exists('composer_manager')) {
  if (function_exists('composer_manager_register_autoloader')) {
    composer_manager_register_autoloader();
  }
  $__openai_sdk_source = 'composer_manager';
}
else {
  $autoload = BACKDROP_ROOT . '/' . backdrop_get_path('module', 'openai') . '/vendor/autoload.php';
  if (file_exists($autoload)) {
    require_once $autoload;
    $__openai_sdk_source = 'module_vendor';
  }
  else {
    $__openai_sdk_source = $__openai_sdk_source ?: 'unknown';
  }
}

use OpenAI\Client as OpenAIClient;
use Symfony\Component\HttpFoundation\StreamedResponse;

class OllamaAdapter implements AIClientInterface {

  /** @var OpenAIClient */
  protected $client;

  /** @var string */
  protected $baseUrl;

  /** @var OpenAIApi|null */
  protected $api;

  /**
   * Constructor.
   *
   * @param string $apiKey
   *   Not used for Ollama, but required by interface. Pass any string.
   * @param OpenAIApi|null $api
   *   Optional parent API wrapper.
   */
  public function __construct($apiKey, $api = NULL) {
    $this->api = $api;
    // Get base URL from central OpenAI providers mapping first, then fall back
    // to module config and finally the localhost default.
    $providers_root = config_get('openai.settings', 'providers') ?: [];
    if (!empty($providers_root['ollama']['ollama_base_url'])) {
      $this->baseUrl = $providers_root['ollama']['ollama_base_url'];
    }
    else {
      $this->baseUrl = config_get('openai_ollama.settings', 'base_url') ?: 'http://localhost:11434';
    }

    // Ensure the URL ends with /v1
    $base_uri = rtrim($this->baseUrl, '/') . '/v1';

    // Ollama doesn't require an API key, but the SDK requires one
    // Use a dummy key that won't be validated
    $dummyKey = 'ollama-local-key';

    $this->client = \OpenAI::factory()
      ->withApiKey($dummyKey)
      ->withBaseUri($base_uri)
      ->make();
  }

  /**
   * {@inheritdoc}
   */
  public function getModels(): array {
    $models = [];
    try {
      $response = $this->client->models()->list();

      foreach ($response->data as $model) {
        $id = $model->id;
        $models[$id] = $id;
      }

      if (!empty($models)) {
        asort($models);
      }
    }
    catch (\Exception $e) {
      watchdog('openai_ollama', 'Failed to fetch models: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
    }

    return $models;
  }


  /**
   * Get models by capability.
   *
   * @param string $capability
   *   The capability to filter by: 'text', 'image', 'vision', 'embeddings'.
   *
   * @return array
   *   Array of model IDs => names.
   */
  public function getModelsByCapability($capability): array {
    return $this->getModels();
  }

  /**
   * Get chat/text generation models.
   *
   * @return array
   *   Array of model IDs => names.
   */
  public function getChatModels(): array {
    return $this->getModelsByCapability('text');
  }

  /**
   * Get image generation models.
   *
   * @return array
   *   Array of model IDs => names.
   */
  public function getImageModels(): array {
    return $this->getModelsByCapability('image');
  }

  /**
   * Get vision models (image input).
   *
   * @return array
   *   Array of model IDs => names.
   */
  public function getVisionModels(): array {
    return $this->getModelsByCapability('vision');
  }

  /**
   * Get embedding models.
   *
   * @return array
   *   Array of model IDs => names.
   */
  public function getEmbeddingModels(): array {
    return $this->getModels();
  }

  /**
   * {@inheritdoc}
   */
  public function completions(string $model, string $prompt, $temperature, $max_tokens = 512, bool $stream_response = FALSE) {
    $params = [
      'model' => $model,
      'prompt' => $prompt,
      'temperature' => (float) $temperature,
      'max_tokens' => (int) $max_tokens,
    ];

    if ($stream_response) {
      $params['stream'] = TRUE;
      $stream = $this->client->completions()->createStreamed($params);
      return $this->handleStreamedResponse($stream);
    }

    $response = $this->client->completions()->create($params);
    return $response->choices[0]->text ?? '';
  }

  /**
   * {@inheritdoc}
   */
  public function chat(string $model, array $messages, $temperature, $max_tokens = 1024, bool $stream_response = FALSE) {
    $params = [
      'model' => $model,
      'messages' => $messages,
      'temperature' => (float) $temperature,
      'max_tokens' => (int) $max_tokens,
    ];

    if ($stream_response) {
      $params['stream'] = TRUE;
      $stream = $this->client->chat()->createStreamed($params);
      return $this->handleStreamedResponse($stream);
    }

    $response = $this->client->chat()->create($params);
    return $response->choices[0]->message->content ?? '';
  }

  /**
   * {@inheritdoc}
   */
  public function images(string $model, string $prompt, string $size, string $response_format, string $quality = 'standard', string $style = 'natural', ?string $output_format = NULL) {
    // Ollama doesn't support image generation through OpenAI-compatible API
    throw new \Exception('Image generation is not supported by Ollama');
  }

  /**
   * {@inheritdoc}
   */
  public function textToSpeech(string $model, string $input, string $voice, string $response_format) {
    // Ollama doesn't support TTS through OpenAI-compatible API
    throw new \Exception('Text-to-speech is not supported by Ollama');
  }

  /**
   * {@inheritdoc}
   */
  public function speechToText(string $model, string $file, string $task = 'transcribe', $temperature = 0.4, string $response_format = 'verbose_json') {
    // Ollama doesn't support STT through OpenAI-compatible API
    throw new \Exception('Speech-to-text is not supported by Ollama');
  }

  /**
   * {@inheritdoc}
   */
  public function moderation(string $input, string $model = 'omni-moderation-latest'): array {
    // Ollama doesn't have built-in moderation
    throw new \Exception('Moderation is not supported by Ollama');
  }

  /**
   * {@inheritdoc}
   */
  public function embedding(string $input, string $model, bool $log = TRUE): array {
    $start_time = microtime(TRUE);
    try {
      $response = $this->client->embeddings()->create([
        'model' => $model,
        'input' => $input,
      ]);

      $result = $response->embeddings[0]->embedding ?? [];
      // Prepare lightweight metadata for logging instead of the full
      // embeddings payload, which can be very large.
      $embedding_count = 0;
      if (isset($response->embeddings) && is_iterable($response->embeddings)) {
        $embedding_count = count($response->embeddings);
      }
      $embedding_dimensions = NULL;
      if ($embedding_count > 0 && isset($response->embeddings[0]->embedding) && is_array($response->embeddings[0]->embedding)) {
        $embedding_dimensions = count($response->embeddings[0]->embedding);
      }
      $response_metadata = [
        'model' => $model,
        'input_length' => function_exists('mb_strlen') ? mb_strlen($input) : strlen($input),
        'embedding_count' => $embedding_count,
        'embedding_dimensions' => $embedding_dimensions,
      ];
      // Note: `method_exists()` accepts an object or a class-name string. If
      // `$this->api` ever holds a class-name string, calling
      // `$this->api->recordLog(...)` will fatal. Require an object here to
      // ensure instance method invocation is safe.
      if (is_object($this->api) && method_exists($this->api, 'recordLog')) {
        $duration = microtime(TRUE) - $start_time;
        $this->api->recordLog('embedding', $model, ['input' => $input], $response_metadata, TRUE, $duration, NULL, !$log);
      }
      return $result;
    }
    catch (\Exception $e) {
      // Also log to the central openai_log if possible.
      // The special-case suppression for "does not support embeddings"
      // currently only affected the watchdog() call. RecordLog() was being
      // invoked before that suppression check, which meant probe errors were
      // still written to the central log when `$log` was TRUE. Apply the same
      // suppression here and require an object for safe method invocation.
      $error_msg = $e->getMessage();
      $is_probe_no_embeddings = (strpos($error_msg, 'does not support embeddings') !== FALSE);
      if (!$is_probe_no_embeddings) {
        if (is_object($this->api) && method_exists($this->api, 'recordLog')) {
          $duration = microtime(TRUE) - $start_time;
          // SECURITY: Logging the raw `input` here can persist end-user content
          // (including PII or secrets) to logs. By default we truncate the
          // logged input to avoid accidental leakage. If you need full input
          // logging, add an explicit opt-in (e.g., `log_full_inputs`) behind
          // protected configuration and audit access to those logs.
          $logged_input = $input;
          if (is_string($input)) {
            $max_log_chars = 200;
            if (function_exists('mb_substr')) {
              $logged_input = mb_strlen($input) > $max_log_chars ? mb_substr($input, 0, $max_log_chars) . '... (truncated)' : $input;
            } else {
              $logged_input = strlen($input) > $max_log_chars ? substr($input, 0, $max_log_chars) . '... (truncated)' : $input;
            }
          }
          $this->api->recordLog('embedding', $model, ['input' => $logged_input], NULL, FALSE, $duration, $error_msg, !$log);
        }
      }
      if ($log) {
        // Specifically suppress watchdog logging for the probing "does not
        // support embeddings" message so probes don't create noise.
        if (!$is_probe_no_embeddings) {
          watchdog('openai_ollama', 'Embedding failed: @error', ['@error' => $error_msg], WATCHDOG_ERROR);
        }
      }
      return [];
    }
  }

  /**
   * Handle streamed responses.
   *
   * @param mixed $stream
   *   The stream from the API.
   *
   * @return StreamedResponse
   *   A Symfony StreamedResponse for output.
   */
  protected function handleStreamedResponse($stream): StreamedResponse {
    return new StreamedResponse(function () use ($stream) {
      foreach ($stream as $chunk) {
        $text = $chunk->choices[0]->delta->content ??
                $chunk->choices[0]->text ?? '';

        if (!empty($text)) {
          echo $text;
          ob_flush();
          flush();
        }
      }
    });
  }
}
