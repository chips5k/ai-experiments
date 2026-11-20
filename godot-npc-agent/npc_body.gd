extends CharacterBody3D

@onready var camera = $NpcCamera
@onready var chat_panel = $"../../../Panel/RichTextLabel"
@onready var viewport: SubViewport = get_parent() as SubViewport
const SPEED = 2.5
const TURN_SPEED = 2.5
const JUMP_VELOCITY = 4.5

var http_request: HTTPRequest
var server_url: String = "http://127.0.0.1:9931/v1/chat/completions"

const system_prompt = "res://SYSTEM_PROMPT.md"
	
var state ="idle"
var actions_queue = []

var request_queue: Array[Dictionary] = []



# Conversation history matching the OpenAI messages format
var chat_history: Array = [
	{
		"role": "system",
		"content": "You are a sentient being living in world, you will receive descriptions of what you can see and must chose an action to perform. actions are move_forward, move_backward, rotate_left, rotate_right"
	}
]

var current_action: String = ""
var action_timer: float = 0.0

func _ready() -> void:
	http_request = HTTPRequest.new()
	add_child(http_request)
	# force renders all the time
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	
	
func request_vision_description():
	# ensure render has occured before sending
	await RenderingServer.frame_post_draw
	var img = viewport.get_texture().get_image()
	img.save_png("res://debug_viewport.png")
	# 1. Convert Image to PNG byte array, then encode as Base64
	var png_bytes: PackedByteArray = img.save_png_to_buffer()
	var base64_img: String = Marshalls.raw_to_base64(png_bytes)
	var image_data_url: String = "data:image/png;base64," + base64_img

	# 2. Build the user payload containing both text and image
	var messages = [
		{
			"role": "system",
			"content": "You are the eyes for a blind person, you will receive images of their field of vision, describe what you see to help them navigate the environment, interpet as three dimensional space via best guess."
		},
		{
		"role": "user",
		"content": [
			{
				"type": "image_url",
				"image_url": {
					"url": image_data_url
				}
			}
		]
	}]
	
	# 4. Construct request payload
	var payload := {
		"messages": messages,
		"temperature": 0.7,
		"max_tokens": 1000,
		"model": "ggml-org/LFM3-Omni-2B-GGUF:Q8_0",
	}
	
	var headers: PackedStringArray = ["Content-Type: application/json"]
	var json_body: String = JSON.stringify(payload)
	
	# 5. Send POST request
	if http_request.request_completed.has_connections():
		http_request.request_completed.disconnect(handle_next_action)
		
	http_request.request_completed.connect(handle_vision_description)
	var err = http_request.request(server_url, headers, HTTPClient.METHOD_POST, json_body)
	if err != OK:
		push_error("Failed to send HTTP Request: " + str(err))


func handle_vision_description(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray):
	
	if response_code == 200:
		var response_text = body.get_string_from_utf8()
		var json = JSON.parse_string(response_text)
		
		request_next_action(json.choices[0].message.content)
		
	else:
		push_error("HTTP Error " + str(response_code) + ": " + body.get_string_from_utf8())


func request_next_action(environment_description: String):

	var user_message = {
		"role": "user",
		"content": [
			{
				"type": "text",
				"text": "Your vision assistant describes what you can see as: " + environment_description + "decide what you will do next. in your response, include your thought process followed by a blank line, and the a final line containing just the action you wish to take: move_forward, move_backward, rotate_left, or rotate_right"
			},
		]
	}
	
	# 3. Append current user message to chat history
	chat_history.append(user_message)
	
	# 4. Construct request payload
	var payload := {
		"messages": chat_history,
		"temperature": 0.7,
		"max_tokens": 1000,
		"model": "ggml-org/gpt-oss-20b-GGUF:MXFP4",
	}
	
	var headers: PackedStringArray = ["Content-Type: application/json"]
	var json_body: String = JSON.stringify(payload)
	
	# 5. Send POST request
	http_request.request_completed.disconnect(handle_vision_description)
	http_request.request_completed.connect(handle_next_action)
	var err = http_request.request(server_url, headers, HTTPClient.METHOD_POST, json_body)
	if err != OK:
		push_error("Failed to send HTTP Request: " + str(err))
	
		
func handle_next_action(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray):
	if response_code == 200:
		var response_text = body.get_string_from_utf8()
		var json = JSON.parse_string(response_text)
		
		if json and json.has("choices") and json["choices"].size() > 0:
			var assistant_response: String = json["choices"][0]["message"]["content"]
			print("LLM Response: ", assistant_response)
			
			var last_line = assistant_response.split("\n")[-1]
			var actions   = last_line.split(",")
			
			for action in actions:
				actions_queue.append(action)
			
			if !actions_queue.is_empty():
				state = "acting"
			
			
			var prev_text = chat_panel.text
			chat_panel.text = assistant_response + "\n\n"
			chat_panel.add_hr()
			chat_panel.append_text("\n\n" + prev_text)
			# Save assistant response back to history for continuous conversation
			chat_history.append({
				"role": "assistant",
				"content": assistant_response
			})
			
		
	else:
		push_error("HTTP Error " + str(response_code) + ": " + body.get_string_from_utf8())
	

func _physics_process(delta: float) -> void:
	
	if state == "idle":
		state = "thinking"
		request_vision_description()
		# Don't return early; let movement/gravity bleed off smoothly

	if state == "acting":
		# If we don't have an active action, grab the next one from the queue
		if current_action == "" and not actions_queue.is_empty():
			current_action = actions_queue.pop_front()
			# Set how long this action should last (e.g., 0.5 seconds per movement step)
			action_timer = 0.5 

		if current_action != "":
			action_timer -= delta
			if action_timer <= 0:
				current_action = "" # Action complete, ready for next

		if actions_queue.is_empty() and current_action == "":
			state = "idle"

	# --- PHYSICS CALCULATIONS (Runs every frame) ---
	
	if not is_on_floor():
		velocity += get_gravity() * delta

	# Jump handling
	if current_action == "Jump" and is_on_floor():
		velocity.y = JUMP_VELOCITY
		current_action = "" # Consume instant action immediately

	# Rotation handling
	var turn := 0.0
	if current_action == "rotate_left": turn = -0.25
	if current_action == "rotate_right": turn = 0.25
	if turn != 0: 
		rotation.y -= turn * TURN_SPEED * delta

	# Movement handling
	var move_input := 0.0
	if current_action == "move_forward": move_input = -0.25
	elif current_action == "move_backward": move_input = 0.25
	
	var direction := (transform.basis * Vector3(0, 0, move_input)).normalized()
	if direction != Vector3.ZERO:
		velocity.x = direction.x * SPEED
		velocity.z = direction.z * SPEED
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)
		velocity.z = move_toward(velocity.z, 0, SPEED)
		
	# Always call move_and_slide so velocity/gravity actually update the position
	move_and_slide()
