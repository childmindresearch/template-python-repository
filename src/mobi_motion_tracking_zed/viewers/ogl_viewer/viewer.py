"""The opengl viewer."""

import array
import ctypes
import math
from threading import Lock

import numpy as np
import OpenGL.GL as GL
import OpenGL.GLUT as GLUT
import pyzed.sl as sl

from mobi_motion_tracking_zed.viewers.cv_viewer import utils

M_PI = 3.1415926

SK_SPHERE_SHADER = """
# version 330 core
layout(location = 0) in vec3 in_Vertex;
layout(location = 1) in vec3 in_Normal;
out vec4 b_color;
out vec3 b_position;
out vec3 b_normal;
uniform mat4 u_mvpMatrix;
uniform vec4 u_color;
uniform vec4 u_pt;
void main() {
   b_color = u_color;
   b_position = in_Vertex;
   b_normal = in_Normal;
   gl_Position =  u_mvpMatrix * (u_pt + vec4(in_Vertex, 1));
}
"""

SK_VERTEX_SHADER = """
# version 330 core
layout(location = 0) in vec3 in_Vertex;
layout(location = 1) in vec3 in_Normal;
out vec4 b_color;
out vec3 b_position;
out vec3 b_normal;
uniform mat4 u_mvpMatrix;
uniform vec4 u_color;
void main() {
   b_color = u_color;
   b_position = in_Vertex;
   b_normal = in_Normal;
   gl_Position =  u_mvpMatrix * vec4(in_Vertex, 1);
}
"""

SK_FRAGMENT_SHADER = """
# version 330 core
in vec4 b_color;
in vec3 b_position;
in vec3 b_normal;
out vec4 out_Color;
void main() {
	vec3 lightPosition = vec3(0, 2, 1);
	float ambientStrength = 0.3;
	vec3 lightColor = vec3(0.75, 0.75, 0.9);
	vec3 ambient = ambientStrength * lightColor;
	vec3 lightDir = normalize(lightPosition - b_position);
	float diffuse = (1 - ambientStrength) * max(dot(b_normal, lightDir), 0.0);
    out_Color = vec4(b_color.rgb * (diffuse + ambient), 1);
}
"""

def generate_color_id(_idx) -> np.ndarray:
    """Generate color id."""
    clr = np.divide(utils.generate_color_id_u(_idx),255.0)
    clr[0], clr[2] = clr[2], clr[0]
    return clr

class Shader:
    """Shader class."""
    def __init__(self, _vs, _fs) -> None:
        """Initialize."""
        self.program_id = GL.glCreateProgram()
        vertex_id = self.compile(GL.GL_VERTEX_SHADER, _vs)
        fragment_id = self.compile(GL.GL_FRAGMENT_SHADER, _fs)

        GL.glAttachShader(self.program_id, vertex_id)
        GL.glAttachShader(self.program_id, fragment_id)
        GL.glBindAttribLocation( self.program_id, 0, "in_vertex")
        GL.glBindAttribLocation( self.program_id, 1, "in_texCoord")
        GL.glLinkProgram(self.program_id)

        if GL.glGetProgramiv(self.program_id, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            info = GL.glGetProgramInfoLog(self.program_id)
            GL.glDeleteProgram(self.program_id)
            GL.glDeleteShader(vertex_id)
            GL.glDeleteShader(fragment_id)
            raise RuntimeError('Error linking program: %s' % (info))
        GL.glDeleteShader(vertex_id)
        GL.glDeleteShader(fragment_id)

    def compile(self, _type, _src) -> any:
        """Compile."""
        try:
            shader_id = GL.glCreateShader(_type)
            if shader_id == 0:
                print("ERROR: shader type {0} does not exist".format(_type))
                exit()

            GL.glShaderSource(shader_id, _src)
            GL.glCompileShader(shader_id)
            if GL.glGetShaderiv(shader_id, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
                info = GL.glGetShaderInfoLog(shader_id)
                GL.glDeleteShader(shader_id)
                raise RuntimeError('Shader compilation failed: %s' % (info))
            return shader_id
        except:
            GL.glDeleteShader(shader_id)
            raise

    def get_program_id(self) -> any:
        """Get program id."""
        return self.program_id

class Simple3DObject:
    """Class that manages simple 3D objects to render with OpenGL."""
    def __init__(self, _is_static) -> None:
        """Initialize."""
        self.vaoID = 0
        self.drawing_type = GL.GL_TRIANGLES
        self.is_static = _is_static
        self.elementbufferSize = 0
        self.is_init = False

        self.vertices = array.array('f')
        self.normals = array.array('f')
        self.indices = array.array('I')

    def __del__(self) -> None:
        """Delete object."""
        self.is_init = False
        if self.vaoID:
            self.vaoID = 0

    def add_vert(self, i_f, limit, height) -> None:
        """Add line."""
        p1 = [i_f, height, -limit]
        p2 = [i_f, height, limit]
        p3 = [-limit, height, i_f]
        p4 = [limit, height, i_f]

        self.add_line(p1, p2)
        self.add_line(p3, p4)

    def add_pt(self, _pts) -> None:  
        """Add a unique point to the list of points."""
        for pt in _pts:
            self.vertices.append(pt)

    def add_normal(self, _normals) -> None:
        """Add a unique normal to the list of normals."""
        for normal in _normals:
            self.normals.append(normal)

    def add_points(self, _pts) -> None:
        """Add a set of points to the list of points and their corresponding color."""
        for i in range(len(_pts)):
            pt = _pts[i]
            self.add_pt(pt)
            current_size_index = int((len(self.vertices)/3))-1
            self.indices.append(current_size_index)
            self.indices.append(current_size_index+1)

    def add_point_clr(self, _pt) -> None:
        """Add a point and its corresponding color to the list of points."""
        self.add_pt(_pt)
        self.add_normal([0.3,0.3,0.3])
        self.indices.append(len(self.indices))

    def add_point_clr_norm(self, _pt,  _norm) -> None:
        """Add a point and its corresponding color and norm to the list of points."""
        self.add_pt(_pt)
        self.add_normal(_norm)
        self.indices.append(len(self.indices))

    def add_line(self, _p1, _p2) -> None:
        """Define a line from two points."""
        self.add_point_clr(_p1)
        self.add_point_clr(_p2)
    
    def add_sphere(self) -> None:
        """Define a sphere.""" 
        m_radius = 0.025
        m_stack_count = 12
        m_sector_count = 12

        for i in range(m_stack_count+1):
            lat0 = M_PI * (-0.5 + (i - 1) / m_stack_count)
            z0 = math.sin(lat0)
            zr0 = math.cos(lat0)

            lat1 = M_PI * (-0.5 + i / m_stack_count)
            z1 = math.sin(lat1)
            zr1 = math.cos(lat1)
            for j in range(m_sector_count):
                lng = 2 * M_PI * (j - 1) / m_sector_count
                x = math.cos(lng)
                y = math.sin(lng)

                v = [m_radius * x * zr0, m_radius * y * zr0, m_radius * z0]
                normal = [x * zr0, y * zr0, z0]
                self.add_point_clr_norm(v, normal)

                v = [m_radius * x * zr1, m_radius * y * zr1, m_radius * z1]
                normal = [x * zr1, y * zr1, z1]
                self.add_point_clr_norm(v, normal)

                lng = 2 * M_PI * j / m_sector_count
                x = math.cos(lng)
                y = math.sin(lng)

                v= [m_radius * x * zr1, m_radius * y * zr1, m_radius * z1]
                normal = [x * zr1, y * zr1, z1]
                self.add_point_clr_norm(v, normal)

                v = [m_radius * x * zr0, m_radius * y * zr0, m_radius * z0]
                normal = [x * zr0, y * zr0, z0]                
                self.add_point_clr_norm(v, normal)

    def push_to_GPU(self) -> None:
        """Push to GPU."""
        if not self.is_init:
            self.vboID = GL.glGenBuffers(3)
            self.is_init = True

        if len(self.vertices):
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vboID[0])
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, 
                len(self.vertices) * self.vertices.itemsize, 
                (GL.GLfloat * len(self.vertices))(*self.vertices), GL.GL_STATIC_DRAW
                )
        
        if len(self.normals):
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vboID[1])
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, 
                len(self.normals) * self.normals.itemsize, 
                (GL.GLfloat * len(self.normals))(*self.normals), GL.GL_STATIC_DRAW
                )

        if len(self.indices):
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER,
                len(self.indices) * self.indices.itemsize,
                (GL.GLuint * len(self.indices))(*self.indices), GL.GL_STATIC_DRAW
                )
            
        self.elementbufferSize = len(self.indices)

    def clear(self) -> None:      
        """Clear vertices, normals, and indices."""  
        self.vertices = array.array('f')
        self.normals = array.array('f')
        self.indices = array.array('I')

    def set_drawing_type(self, _type) -> None:
        """Set drawing type."""
        self.drawing_type = _type

    def draw(self) -> None:
        """Draw."""
        if (self.elementbufferSize > 0) and self.is_init:            
            GL.glEnableVertexAttribArray(0)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vboID[0])
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            
            GL.glEnableVertexAttribArray(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vboID[1])
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.vboID[2])
            GL.glDrawElements(
                self.drawing_type, 
                self.elementbufferSize, 
                GL.GL_UNSIGNED_INT, 
                None
                )
            
            GL.glDisableVertexAttribArray(0)
            GL.glDisableVertexAttribArray(1)

class Skeleton:
    """Dataclass for ZED Skeleton."""
    def __init__(self, _body_format = sl.BODY_FORMAT.BODY_18) -> None:
        """Initialize."""
        self.clr = [0,0,0,1]
        self.kps = []
        self.joints = Simple3DObject(False)
        self.Z = 1
        self.body_format = _body_format

    def createSk(self, obj, BODY_PARTS, BODY_BONES) -> None:
        """Create the skeleton."""
        for bone in BODY_BONES:
            kp_1 = obj.keypoint[bone[0].value]
            kp_2 = obj.keypoint[bone[1].value]
            if math.isfinite(kp_1[0]) and math.isfinite(kp_2[0]):
                self.joints.add_line(kp_1, kp_2)

        for part in range(len(BODY_PARTS)-1):    # -1 to avoid LAST
            kp = obj.keypoint[part]
            norm = np.linalg.norm(kp)
            if math.isfinite(norm):
                self.kps.append(kp)

    def set(self, obj) -> None:
        """Set the skeleton."""
        self.joints.set_drawing_type(GL.GL_LINES)
        self.clr = generate_color_id(obj.id)
        self.Z = abs(obj.position[2])
        # Draw skeletons
        if obj.keypoint.size > 0:
            if self.body_format == sl.BODY_FORMAT.BODY_18:
                self.createSk(obj, sl.BODY_18_PARTS, sl.BODY_18_BONES)
            elif self.body_format == sl.BODY_FORMAT.BODY_34:
                self.createSk(obj, sl.BODY_34_PARTS, sl.BODY_34_BONES)
            elif self.body_format == sl.BODY_FORMAT.BODY_38:
                self.createSk(obj, sl.BODY_38_PARTS, sl.BODY_38_BONES)

    def push_to_GPU(self) -> None:
        """Push to GPU."""
        self.joints.push_to_GPU()

    def draw(self, shader_sk_clr, sphere, shader_mvp, projection) -> None:
        """Draw the skeleton."""
        GL.glUniform4f(shader_sk_clr, self.clr[0],self.clr[1],self.clr[2],self.clr[3])
        line_w = (20. / self.Z)
        GL.glLineWidth(line_w)
        self.joints.draw()

    def drawKPS(self, shader_clr, sphere, shader_pt) -> None:
        """Draw keypoints."""
        GL.glUniform4f(shader_clr, self.clr[0],self.clr[1],self.clr[2],self.clr[3])
        for k in self.kps:
            GL.glUniform4f(shader_pt, k[0],k[1],k[2], 1)
            sphere.draw()

IMAGE_FRAGMENT_SHADER = """
# version 330 core
in vec2 UV;
out vec4 color;
uniform sampler2D texImage;
void main() {
    vec2 scaler =vec2(UV.x,1.f - UV.y);
    vec3 rgbcolor = vec3(texture(texImage, scaler).zyx);
    vec3 color_rgb = pow(rgbcolor, vec3(1.65f));
    color = vec4(color_rgb,1.f);
}
"""

IMAGE_VERTEX_SHADER = """
# version 330
layout(location = 0) in vec3 vert;
out vec2 UV;
void main() {
    UV = (vert.xy+vec2(1.f,1.f))*.5f;
    gl_Position = vec4(vert, 1.f);
}
"""

class ImageHandler:
    """Class that manages the image stream to render with OpenGL."""
    def __init__(self) -> None:
        """Initialize."""
        self.tex_id = 0
        self.image_tex = 0
        self.quad_vb = 0
        self.is_called = 0

    def close(self) -> None:
        """Close image."""
        if self.image_tex:
            self.image_tex = 0

    def initialize(self, _res) -> None:    
        """Initialize image."""
        self.shader_image = Shader(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER)
        self.tex_id = GL.glGetUniformLocation(
            self.shader_image.get_program_id(), 
            "texImage"
            )

        g_quad_vertex_buffer_data = np.array([-1, -1, 0,
                                               1, -1, 0,
                                              -1, 1, 0,
                                              -1, 1, 0,
                                               1, -1, 0,
                                               1, 1, 0], 
                                               np.float32)

        self.quad_vb = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.quad_vb)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, g_quad_vertex_buffer_data.nbytes,
                     g_quad_vertex_buffer_data, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        # Create and populate the texture
        GL.glEnable(GL.GL_TEXTURE_2D)

        # Generate a texture name
        self.image_tex = GL.glGenTextures(1)
        
        # Select the created texture
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_tex)
        
        # Set the texture minification and magnification filters
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        
        # Fill the texture with an image
        # None means reserve texture memory, but texels are undefined
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, _res.width, _res.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        
        # Unbind the texture
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)   

    def push_new_image(self, _zed_mat) -> None:
        """Push new image."""
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_tex)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, _zed_mat.get_width(), _zed_mat.get_height(), GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,  ctypes.c_void_p(_zed_mat.get_pointer()))
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)            

    def draw(self) -> None:
        """Draw image."""
        GL.glUseProgram(self.shader_image.get_program_id())
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_tex)
        GL.glUniform1i(self.tex_id, 0)

        GL.glEnableVertexAttribArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.quad_vb)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glDisableVertexAttribArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)            
        GL.glUseProgram(0)

class GLViewer:
    """Class that manages input events, window and OpenGL rendering pipeline."""
    def __init__(self) -> None:
        """Initialize."""
        self.available = False
        self.bodies = []
        self.mutex = Lock()
        # Create the rendering camera
        self.projection = array.array('f')
        self.basic_sphere = Simple3DObject(True)
        # Show tracked objects only
        self.is_tracking_on = False
        self.body_format = sl.BODY_FORMAT.BODY_18

    def init(self, _params, _is_tracking_on, _body_format) -> None:
        """Initiate viewer.""" 
        GLUT.glutInit()
        wnd_w = GLUT.glutGet(GLUT.GLUT_SCREEN_WIDTH)
        wnd_h = GLUT.glutGet(GLUT.GLUT_SCREEN_HEIGHT)
        width = (int)(wnd_w*0.9)
        height = (int)(wnd_h*0.9)
     
        GLUT.glutInitWindowSize(width, height)
        # The window opens at the upper left corner of the screen
        GLUT.glutInitWindowPosition((int)(wnd_w*0.05), (int)(wnd_h*0.05)) 
        GLUT.glutInitDisplayMode(GLUT.GLUT_DOUBLE | GLUT.GLUT_SRGB)
        GLUT.glutCreateWindow("ZED Body Tracking")
        GL.glViewport(0, 0, width, height)

        GLUT.glutSetOption(GLUT.GLUT_ACTION_ON_WINDOW_CLOSE,
                      GLUT.GLUT_ACTION_CONTINUE_EXECUTION)

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)

        # Compile and create the shader for 3D objects
        self.shader_sk_image = Shader(SK_VERTEX_SHADER, SK_FRAGMENT_SHADER)
        self.shader_sk_MVP = GL.glGetUniformLocation(
            self.shader_sk_image.get_program_id(), 
            "u_mvpMatrix"
            )
        self.shader_sk_clr = GL.glGetUniformLocation(
            self.shader_sk_image.get_program_id(), 
            "u_color"
            )
        
        self.shader_sphere_image = Shader(SK_SPHERE_SHADER, SK_FRAGMENT_SHADER)
        self.shader_sphere_MVP = GL.glGetUniformLocation(
            self.shader_sphere_image.get_program_id(), 
            "u_mvpMatrix"
            )
        self.shader_sphere_clr = GL.glGetUniformLocation(
            self.shader_sphere_image.get_program_id(), 
            "u_color"
            )
        self.shader_sphere_pt = GL.glGetUniformLocation(
            self.shader_sphere_image.get_program_id(), 
            "u_pt"
            )

        self.set_render_camera_projection(_params, 0.1, 200)

        self.floor_plane_set = False

        self.is_tracking_on = _is_tracking_on

        self.basic_sphere.add_sphere()        
        self.basic_sphere.set_drawing_type(GL.GL_QUADS)
        self.basic_sphere.push_to_GPU()

        # Register the drawing function with GLUT
        GLUT.glutDisplayFunc(self.draw_callback)
        # Register the function called when nothing happens
        GLUT.glutIdleFunc(self.idle)   

        GLUT.glutKeyboardFunc(self.keyPressedCallback)
        # Register the closing function
        GLUT.glutCloseFunc(self.close_func)

        self.available = True
        self.body_format = _body_format

    def set_floor_plane_equation(self, _eq) -> None:
        """Set floor plane equation."""
        self.floor_plane_set = True
        self.floor_plane_eq = _eq

    def set_render_camera_projection(self, _params, _znear, _zfar) -> None:
        """Set render camera projection."""
        # Just slightly move up the ZED camera FOV to make a small black border
        fov_y = (_params.v_fov + 0.5) * M_PI / 180
        fov_x = (_params.h_fov + 0.5) * M_PI / 180

        self.projection.append(1 / math.tan(fov_x * 0.5) )  # Horizontal FoV.
        self.projection.append(0)
        # Horizontal offset.
        self.projection.append( 
            2 * ((_params.image_size.width - _params.cx) / _params.image_size.width) - 1
            )
        self.projection.append(0)

        self.projection.append(0)
        self.projection.append(1 / math.tan(fov_y * 0.5))  # Vertical FoV.
        # Vertical offset.
        self.projection.append(-(2*(
                (_params.image_size.height - _params.cy)/_params.image_size.height)-1))
        self.projection.append(0)

        self.projection.append(0)
        self.projection.append(0)
        # Near and far planes.
        self.projection.append(-(_zfar + _znear) / (_zfar - _znear))
        # Near and far planes.
        self.projection.append(-(2 * _zfar * _znear) / (_zfar - _znear))

        self.projection.append(0)
        self.projection.append(0)
        self.projection.append(-1)
        self.projection.append(0)
        

    def is_available(self) -> None:
        """Check if main is available."""
        if self.available:
            GLUT.glutMainLoopEvent()
        return self.available

    def render_object(self, _object_data: sl.ObjectData) -> any:
        """Render zed object."""
        if self.is_tracking_on:
            return (_object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK)
        else:
            return (
                _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK 
                or 
                _object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF
                )

    def update_view(self, _image, _bodies: sl.Bodies) -> None:
        """Update view with image and bodies."""
        self.mutex.acquire()

        # Clear objects
        self.bodies.clear()
        # Only show tracked objects
        for _body in _bodies.body_list:
            if self.render_object(_body):
                current_sk = Skeleton(self.body_format)
                current_sk.set(_body)
                self.bodies.append(current_sk)
        self.mutex.release()

    def idle(self) -> None:
        """Idle mode."""
        if self.available:
            GLUT.glutPostRedisplay()

    def exit(self) -> None:
        """Exit view."""      
        if self.available:
            self.available = False

    def close_func(self) -> None:
        """Close function.""" 
        if self.available:
            self.available = False     

    def keyPressedCallback(self, key, x, y) -> None:
        """Key callback close function."""
        if ord(key) == 113 or ord(key) == 27:
            self.close_func() 

    def draw_callback(self) -> None:
        """Draw callback."""
        if self.available:
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            self.mutex.acquire()
            self.update()
            self.draw()
            self.mutex.release()  

            GLUT.glutSwapBuffers()
            GLUT.glutPostRedisplay()

    def update(self) -> None:
        """Update Bodies."""
        for body in self.bodies:
            body.push_to_GPU()

    def draw(self) -> None:
        """Draw display."""
        GL.glUseProgram(self.shader_sk_image.get_program_id())
        GL.glUniformMatrix4fv(
            self.shader_sk_MVP, 
            1, 
            GL.GL_TRUE, 
            (GL.GLfloat * len(self.projection))(*self.projection)
            )
        
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        for body in self.bodies:
            body.draw(
                self.shader_sphere_clr, 
                self.basic_sphere, 
                self.shader_sphere_MVP, 
                self.projection)
        GL.glUseProgram(0)

        GL.glUseProgram(self.shader_sphere_image.get_program_id())
        GL.glUniformMatrix4fv(
            self.shader_sphere_MVP, 
            1, 
            GL.GL_TRUE,  
            (GL.GLfloat * len(self.projection))(*self.projection)
            )
        for body in self.bodies:
            body.drawKPS(
                self.shader_sphere_clr, 
                self.basic_sphere, 
                self.shader_sphere_pt
                )
        GL.glUseProgram(0)
