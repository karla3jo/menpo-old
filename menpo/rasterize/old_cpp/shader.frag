#version 330

uniform sampler2D texture_image;
smooth in vec2 tcoord;
smooth in vec3 color;

layout(location = 0) out vec3 outputColor;
layout(location = 1) out vec3 outputCoord;

void main()
{
   outputColor = texture(texture_image, tcoord).rgb;
   outputCoord = color;
}
